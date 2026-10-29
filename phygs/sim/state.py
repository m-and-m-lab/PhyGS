# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab state restoration and position-target execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from phygs.runtime.episode import JointTargets
from phygs.scene.assets import SCENE_BODIES, SceneBodies
from phygs.sim.fabric import FabricTransformWrites

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext


class IsaacLabWorld:
    """Own a registered rigid/articulated scene and its initial physical snapshot.

    Construct after simulation initialization. All movable entities must be registered
    with the scene. This adapter supports one environment, explicit-PD gain updates, and handing
    explicit-actuator joints that support it to PhysX joint drives.
    """

    def __init__(
        self,
        sim: SimulationContext,
        scene: InteractiveScene,
        *,
        robot_name: str = "robot",
        physics_sensors: tuple[str, ...] = (),
        render_during_step: bool = True,
        gravity_compensation_joints: tuple[str, ...] = (),
        fabric_writes: FabricTransformWrites | None = None,
    ) -> None:
        if scene.num_envs != 1:
            raise ValueError("IsaacLabWorld currently requires exactly one environment.")
        if scene.deformable_objects or scene.surface_grippers or set(scene.rigid_object_collections) - {SCENE_BODIES}:
            raise ValueError(
                "This lifecycle supports registered articulations, rigid objects and the scene-body collection only."
            )
        self._physics_sensors = tuple(scene.sensors[name] for name in physics_sensors)
        self._render_during_step = render_during_step
        self._fabric_writes = fabric_writes
        self._sim = sim
        self._scene = scene
        self._robot = scene[robot_name]
        self.physics_dt = sim.get_physics_dt()
        self.joint_names = tuple(self._robot.joint_names)
        self._joint_indices = {name: index for index, name in enumerate(self.joint_names)}
        self._gravity_joints = [self._joint_indices[name] for name in gravity_compensation_joints]
        self._gravity_dofs = [index + self._robot.num_base_dofs for index in self._gravity_joints]
        self._applied_positions: dict[str, float] = {}
        self._applied_velocities: dict[str, float] = {}
        self._initial_state = scene.get_state()
        # IsaacLab's scene state leaves out collections: the scene bodies' own snapshot.
        self.scene_bodies = SceneBodies(scene) if SCENE_BODIES in scene.rigid_object_collections else None
        self._initial_bodies = (
            (self.scene_bodies.poses(), self.scene_bodies.velocities()) if self.scene_bodies is not None else None
        )
        self._initial_gains = {
            (name, key): (actuator.stiffness.clone(), actuator.damping.clone())
            for name, asset in scene.articulations.items()
            for key, actuator in asset.actuators.items()
        }
        self._gain_joints = {}
        self._driven: set[str] = set()  # Joints on PhysX joint drives.
        for actuator in self._robot.actuators.values():
            if not actuator.is_implicit_model:
                for local, index in enumerate(actuator.joint_indices):
                    self._gain_joints[self.joint_names[int(index)]] = (actuator, local)

        if set(gravity_compensation_joints) & self._gain_joints.keys():
            raise ValueError("Gravity feedforward requires implicit position actuators.")

    def restore(self) -> None:
        """Restore registered poses [m, quaternion xyzw], velocities and joint state."""
        self._applied_positions.clear()
        self._applied_velocities.clear()
        self._release_drives(set(self._driven))
        if self._fabric_writes is not None:  # Settling and the first capture render the restored poses.
            self._fabric_writes.set(True)
        for (name, key), (stiffness, damping) in self._initial_gains.items():
            actuator = self._scene[name].actuators[key]
            actuator.stiffness.copy_(stiffness)
            actuator.damping.copy_(damping)
        self._scene.reset()
        self._scene.reset_to(self._initial_state)
        if self.scene_bodies is not None:
            self.scene_bodies.clear_held_forces()
            self.scene_bodies.write_all(*self._initial_bodies)
        for articulation in self._scene.articulations.values():
            # Physical velocity is snapshot state, not a desired PD velocity.
            zero = torch.zeros_like(articulation.data.joint_pos.torch)
            articulation.set_joint_velocity_target_index(target=zero)
            articulation.set_joint_effort_target_index(target=zero)
        self._scene.write_data_to_sim()
        self._scene.update(0.0)

    def positions(self) -> dict[str, float]:
        """Return measured Spot revolute-joint positions [rad]."""
        values = self._robot.data.joint_pos.torch[0].tolist()
        return dict(zip(self.joint_names, values, strict=True))

    def velocities(self) -> dict[str, float]:
        """Return measured revolute-joint velocities [rad/s]."""
        return dict(zip(self.joint_names, self._robot.data.joint_vel.torch[0].tolist(), strict=True))

    def apply(self, targets: dict[str, float] | JointTargets) -> None:
        """Write joint position [rad] and velocity [rad/s] targets without stepping."""
        velocities = {}
        if isinstance(targets, JointTargets):
            targets.validate(tuple(targets.positions))
            unsupported = (set(targets.stiffness) | set(targets.drive_stiffness)) - self._gain_joints.keys()
            if unsupported:
                raise ValueError(f"Gain changes require explicit PD actuators: {sorted(unsupported)}")
            self._release_drives(set(targets.stiffness) & self._driven)
            for joint, stiffness in targets.stiffness.items():
                actuator, index = self._gain_joints[joint]
                actuator.stiffness[:, index] = stiffness
                actuator.damping[:, index] = targets.damping[joint]
            if targets.drive_stiffness:
                self._drive(targets.drive_stiffness, targets.drive_damping)
            velocities = targets.velocities
            targets = targets.positions
        desired_velocities = {name: velocities.get(name, 0.0) for name in targets}
        changed = {n: v for n, v in desired_velocities.items() if self._applied_velocities.get(n) != v}
        if changed:
            self._robot.set_joint_velocity_target_index(
                target=torch.tensor([list(changed.values())], device=self._robot.device, dtype=torch.float32),
                joint_ids=[self._joint_indices[n] for n in changed],
            )
            self._applied_velocities.update(changed)
        # The articulation holds targets between writes; actuators still execute every tick.
        targets = {name: value for name, value in targets.items() if self._applied_positions.get(name) != value}
        if targets:
            indices = [self._joint_indices[name] for name in targets]
            values = torch.tensor([list(targets.values())], device=self._robot.device, dtype=torch.float32)
            self._robot.set_joint_position_target_index(target=values, joint_ids=indices)
            self._applied_positions.update(targets)

    def _drive(self, stiffness: dict[str, float], damping: dict[str, float]) -> None:
        """Hand joints to PhysX joint drives with these gains [N·m/rad, N·m·s/rad]."""
        for joint in stiffness:
            actuator, index = self._gain_joints[joint]
            if not hasattr(actuator, "set_physx_drive"):
                raise ValueError(f"The actuator of {joint} cannot hand it to a PhysX joint drive.")
            actuator.set_physx_drive([index], True)
        self._write_drive_gains(list(stiffness), list(stiffness.values()), [damping[j] for j in stiffness])
        self._driven.update(stiffness)

    def _release_drives(self, joints: set[str]) -> None:
        """Return joints from PhysX joint drives to their explicit actuators (zero drive gains)."""
        if not joints:
            return
        for joint in joints:
            actuator, index = self._gain_joints[joint]
            actuator.set_physx_drive([index], False)
        self._write_drive_gains(list(joints), [0.0] * len(joints), [0.0] * len(joints))
        self._driven -= joints

    def _write_drive_gains(self, joints: list[str], stiffness: list[float], damping: list[float]) -> None:
        ids = [self._joint_indices[name] for name in joints]
        stiffness, damping = (torch.tensor([values], device=self._robot.device) for values in (stiffness, damping))
        self._robot.write_joint_stiffness_to_sim_index(stiffness=stiffness, joint_ids=ids)
        self._robot.write_joint_damping_to_sim_index(damping=damping, joint_ids=ids)

    def step(self) -> None:
        """Advance physics by the configured timestep [s] and refresh state."""
        if self._gravity_joints:
            efforts = self._robot.data.gravity_compensation_forces.torch[:, self._gravity_dofs]
            self._robot.set_joint_effort_target_index(target=efforts, joint_ids=self._gravity_joints)
        self._scene.write_data_to_sim()
        if self.scene_bodies is not None:
            self.scene_bodies.apply_held_forces()
        render = (
            self._render_during_step
            and self._sim.is_rendering
            and (self._sim.get_physics_step_count() + 1) % self._sim.cfg.render_interval == 0
        )
        self._sim.step(render=render)
        self._scene.update(self.physics_dt)
        # Finite-difference sensors require every physical sample, including reset settling.
        for sensor in self._physics_sensors:
            _ = sensor.data
