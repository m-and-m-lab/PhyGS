# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register existing USD rigid bodies with the shared scene/reset lifecycle.

All of a scene's own rigid bodies are registered as one ``RigidObjectCollection``
(:data:`SCENE_BODIES` in the scene), not one ``RigidObject`` each: IsaacLab clears every
asset's force buffers every physics tick, which for a house's 310 bodies cost 3.6 ms per tick
(2026-10-15). PhysX simulates the same bodies either way. :class:`SceneBodies` reads and writes
them by name or USD path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Scene entry of the collection holding a scene's own rigid bodies.
SCENE_BODIES = "scene_bodies"


def register_scene_bodies(stage: Usd.Stage, config: InteractiveSceneCfg) -> tuple[str, ...]:
    """Register every existing rigid body of the stage before spawning the robot, at its authored pose.

    The bodies form one collection, :data:`SCENE_BODIES`; each is named by its USD path, in stage
    order. Returns the names. Scene articulations need an explicit articulation configuration and
    are rejected here. Ordinary joint-connected rigid bodies (drawers, doors) are supported.
    """
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

    bodies = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError(f"Scene articulation requires an explicit configuration: {prim.GetPath()}")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(str(prim.GetPath()))
    if bodies:
        setattr(
            config,
            SCENE_BODIES,
            RigidObjectCollectionCfg(
                rigid_objects={path: RigidObjectCfg(prim_path=path, spawn=None) for path in bodies}
            ),
        )
    return tuple(bodies)


class SceneBodies:
    """The scene's own rigid bodies (:func:`register_scene_bodies`), by USD path.

    Poses are world link poses [m, quaternion xyzw]; velocities are world center-of-mass
    velocities [m/s, rad/s]. A held force acts on one body's center of mass every physics tick
    until changed (:meth:`hold_force`); it is applied to that body only.
    """

    def __init__(self, scene: InteractiveScene) -> None:
        self._bodies = scene[SCENE_BODIES]
        self.names: tuple[str, ...] = tuple(self._bodies.body_names)
        self._index = {name: index for index, name in enumerate(self.names)}
        self._forces: dict[int, torch.Tensor] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def index(self, name: str) -> int:
        """The body's index in the collection; ``name`` is its USD path."""
        if name not in self._index:
            raise KeyError(f"Not a registered scene body: {name}")
        return self._index[name]

    def pose(self, name: str) -> torch.Tensor:
        """World link pose [x, y, z, qx, qy, qz, qw] (m) of one body."""
        return self._bodies.data.body_link_pose_w.torch[0, self.index(name)].clone()

    def velocity(self, name: str) -> torch.Tensor:
        """World center-of-mass velocity [vx, vy, vz, wx, wy, wz] (m/s, rad/s) of one body."""
        return self._bodies.data.body_com_vel_w.torch[0, self.index(name)].clone()

    def poses(self) -> torch.Tensor:
        """World link poses of all bodies, shape (bodies, 7), in :attr:`names` order."""
        return self._bodies.data.body_link_pose_w.torch[0].clone()

    def velocities(self) -> torch.Tensor:
        """World center-of-mass velocities of all bodies, shape (bodies, 6)."""
        return self._bodies.data.body_com_vel_w.torch[0].clone()

    def write_pose(self, name: str, pose: torch.Tensor | list[float]) -> None:
        """Place one body at a world link pose [m, xyzw]; other bodies are not written."""
        pose = torch.as_tensor(pose, dtype=torch.float32, device=self._bodies.device).reshape(1, 1, 7)
        self._bodies.write_body_link_pose_to_sim_index(body_poses=pose, body_ids=[self.index(name)])

    def write_velocity(self, name: str, velocity: torch.Tensor | list[float]) -> None:
        """Set one body's world center-of-mass velocity [m/s, rad/s]."""
        velocity = torch.as_tensor(velocity, dtype=torch.float32, device=self._bodies.device).reshape(1, 1, 6)
        self._bodies.write_body_com_velocity_to_sim_index(body_velocities=velocity, body_ids=[self.index(name)])

    def write_all(self, poses: torch.Tensor, velocities: torch.Tensor) -> None:
        """Set every body's link pose (bodies, 7) [m, xyzw] and center-of-mass velocity (bodies, 6)."""
        count = len(self.names)
        self._bodies.write_body_link_pose_to_sim_index(body_poses=poses.reshape(1, count, 7).clone())
        self._bodies.write_body_com_velocity_to_sim_index(body_velocities=velocities.reshape(1, count, 6).clone())

    def hold_force(self, name: str, force: torch.Tensor | list[float] | None) -> None:
        """Hold a world force [N] on one body's center of mass every tick; None releases it.

        IsaacLab's collection force buffers would write all bodies every tick (waking every
        sleeping body), so held forces go to the named bodies only.
        """
        index = self.index(name)
        if force is None:
            self._forces.pop(index, None)
            return
        force = torch.as_tensor(force, dtype=torch.float32).reshape(3)
        if not torch.isfinite(force).all():
            raise ValueError("A held force must be finite.")
        self._forces[index] = force

    def apply_held_forces(self) -> None:
        """Apply the held forces for the coming physics step (the world calls this every tick)."""
        if not self._forces:
            return
        import warp as wp

        count, device = len(self.names), self._bodies.device
        indices = sorted(self._forces)
        forces = np.zeros((count, 3), dtype=np.float32)
        for index in indices:
            forces[index] = self._forces[index].numpy()
        positions = self._bodies.data.body_com_pos_w.torch[0].cpu().numpy().astype(np.float32)
        self._bodies.root_view.apply_forces_and_torques_at_position(
            force_data=wp.array(forces.reshape(-1), dtype=wp.float32, device=device),
            torque_data=None,
            position_data=wp.array(positions.reshape(-1), dtype=wp.float32, device=device),
            indices=self._bodies._env_body_ids_to_view_ids(torch.tensor([0]), torch.tensor(indices), device=device),
            is_global=True,
        )

    def clear_held_forces(self) -> None:
        self._forces.clear()

    def body(self, name: str) -> SceneBody:
        """One body as a read-only view with a ``RigidObject``-like ``data`` (root pose, position, velocity)."""
        return SceneBody(self, self.index(name))


class _Field:
    """A ``(1, n)`` tensor under ``.torch``, as IsaacLab's ProxyArray presents asset data."""

    def __init__(self, value: torch.Tensor) -> None:
        self.torch = value.unsqueeze(0)


class _BodyData:
    def __init__(self, bodies: SceneBodies, index: int) -> None:
        self._data, self._index = bodies._bodies.data, index

    @property
    def root_pose_w(self) -> _Field:
        return _Field(self._data.body_link_pose_w.torch[0, self._index].clone())

    root_link_pose_w = root_pose_w

    @property
    def root_pos_w(self) -> _Field:
        return _Field(self._data.body_link_pose_w.torch[0, self._index, :3].clone())

    @property
    def root_com_vel_w(self) -> _Field:
        return _Field(self._data.body_com_vel_w.torch[0, self._index].clone())

    @property
    def root_lin_vel_w(self) -> _Field:
        return _Field(self._data.body_com_vel_w.torch[0, self._index, :3].clone())


class SceneBody:
    """Read access to one scene body, as ``asset.data.root_pose_w.torch[0]`` reads a ``RigidObject``."""

    def __init__(self, bodies: SceneBodies, index: int) -> None:
        self.name = bodies.names[index]
        self.data = _BodyData(bodies, index)


def scene_body_pose(scene: InteractiveScene, key: str) -> torch.Tensor:
    """World link pose [m, xyzw] of a movable scene entity: a rigid-object asset, or a scene body path."""
    if key in scene.rigid_objects:
        return scene[key].data.root_pose_w.torch[0]
    if SCENE_BODIES in scene.rigid_object_collections:
        bodies = scene[SCENE_BODIES]
        if key in bodies.body_names:
            return bodies.data.body_link_pose_w.torch[0, bodies.body_names.index(key)]
    raise KeyError(f"Not a registered rigid object or scene body: {key}")


def has_scene_body(scene: InteractiveScene, key: str) -> bool:
    """Whether ``key`` names a rigid-object asset or a registered scene body."""
    if key in scene.rigid_objects:
        return True
    collections = scene.rigid_object_collections
    return SCENE_BODIES in collections and key in collections[SCENE_BODIES].body_names


def validate_spawn_disk(
    stage: Usd.Stage, center: tuple[float, float], radius: float, height: tuple[float, float]
) -> None:
    """Conservatively validate a declared free disk [m] in a height band [m].

    Uses collider world AABBs only during task setup, before robot spawning.
    This privileged check is not an observation or a mapping input.
    """
    if not all(map(math.isfinite, (*center, radius, *height))) or radius <= 0 or height[0] >= height[1]:
        raise ValueError("Invalid spawn disk geometry")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lower, upper = bounds.GetMin(), bounds.GetMax()
        if upper[2] <= height[0] or lower[2] >= height[1]:
            continue
        distance = math.hypot(*(max(lower[i] - center[i], center[i] - upper[i], 0) for i in (0, 1)))
        if distance <= radius:
            raise ValueError(f"Declared free spawn disk overlaps collider {prim.GetPath()}")
