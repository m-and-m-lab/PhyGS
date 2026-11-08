# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A scene's articulated furniture, read from the stage: which bodies move only through joints.

A scene may tie a cabinet and its doors and drawers into one PhysX articulation (an
``ArticulationRootAPI``, usually on the fixed joint that mounts the cabinet to the world; the
SimReady kitchen of bear_loop_house 1.2.0 does this for its 32 storage units). Such a link is still a
rigid body of the scene, read like any other, but PhysX places it from its joint positions only: a
pose written to it, before the start or later, is ignored. So a drawer opens before the start by its
joint state (:func:`author_joint_position`), and a link is placed later by setting its joint
position (:func:`joint_position` gives it from the two bodies' poses).

Read and author only (testable without Isaac Sim); ``phygs.scene.assets`` writes through PhysX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pxr import Sdf, Usd, UsdPhysics
from scipy.spatial.transform import Rotation

AXES = "XYZ"
# Joint state as PhysX reads it at the start (PhysxSchema's multiple-apply PhysicsJointStateAPI):
# position in metres along a prismatic joint ("linear"), degrees about a revolute joint ("angular").
STATE_INSTANCE = {"prismatic": "linear", "revolute": "angular"}


@dataclass(frozen=True)
class Joint:
    """One joint between two links of an articulation, with its frames on each body."""

    path: str
    kind: str  # "prismatic" or "revolute".
    axis: int  # 0, 1 or 2: the joint frame's axis it moves along or about.
    body0: str
    body1: str
    frame0: np.ndarray  # 4x4 joint frame in body0's frame.
    frame1: np.ndarray  # 4x4 joint frame in body1's frame.


@dataclass(frozen=True)
class Articulation:
    """One articulation: its root prim, its base link (``fixed``: mounted to the world), its links and
    the joints that move them."""

    root: str
    base: str
    fixed: bool  # The base is fixed to the world (the root is a fixed joint with no body0).
    links: tuple[str, ...]
    joints: tuple[Joint, ...]


def articulations(stage: Usd.Stage, bodies: set[str] | None = None) -> list[Articulation]:
    """The stage's articulations whose base is one of ``bodies`` (all of them when None), in stage order.

    Links are the base and every body reached from it through joints (body0 to body1). Prismatic
    and revolute joints move links, a fixed joint carries its link along; any other kind raises.
    """
    edges: dict[str, list[Usd.Prim]] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if parents and children:
                edges.setdefault(str(parents[0]), []).append(prim)
    found = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = joint.GetBody1Rel().GetTargets()
            if not targets:
                raise ValueError(f"Articulation root joint {prim.GetPath()} has no body1.")
            base, fixed = str(targets[0]), prim.IsA(UsdPhysics.FixedJoint) and not joint.GetBody0Rel().GetTargets()
        else:
            base, fixed = str(prim.GetPath()), False
        if bodies is not None and base not in bodies:
            continue
        links, joints, todo = [base], [], [base]
        while todo:
            parent = todo.pop(0)
            for joint_prim in edges.get(parent, []):
                child = str(UsdPhysics.Joint(joint_prim).GetBody1Rel().GetTargets()[0])
                if not joint_prim.IsA(UsdPhysics.FixedJoint):  # A fixed link has no joint position.
                    joints.append(_joint(joint_prim))
                if child not in links:
                    links.append(child)
                    todo.append(child)
        found.append(Articulation(str(prim.GetPath()), base, fixed, tuple(links), tuple(joints)))
    return found


def articulated_joints(stage: Usd.Stage) -> set[str]:
    """Paths of the joints inside the stage's articulations: these move their link, a pose does not."""
    return {joint.path for articulation in articulations(stage) for joint in articulation.joints}


def joint_position(joint: Joint, world_from_body0: np.ndarray, world_from_body1: np.ndarray) -> float:
    """A joint's position from its two bodies' world poses (4x4): metres along a prismatic joint,
    radians about a revolute joint (PhysX's joint coordinates)."""
    frame0, frame1 = world_from_body0 @ joint.frame0, world_from_body1 @ joint.frame1
    if joint.kind == "prismatic":
        return float(np.dot(frame1[:3, 3] - frame0[:3, 3], frame0[:3, joint.axis]))
    return float(Rotation.from_matrix(frame0[:3, :3].T @ frame1[:3, :3]).as_rotvec()[joint.axis])


def author_joint_position(stage: Usd.Stage, joint: Joint, position: float) -> None:
    """Start an articulated joint at ``position`` (metres, or radians for a revolute joint): its
    joint state, which PhysX reads at the start instead of the link's authored pose."""
    instance = STATE_INSTANCE[joint.kind]
    value = position if joint.kind == "prismatic" else float(np.degrees(position))
    prim = stage.GetPrimAtPath(joint.path)
    prim.AddAppliedSchema(f"PhysicsJointStateAPI:{instance}")
    prim.CreateAttribute(f"state:{instance}:physics:position", Sdf.ValueTypeNames.Float).Set(value)
    prim.CreateAttribute(f"state:{instance}:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)


def _joint(prim: Usd.Prim) -> Joint:
    if prim.IsA(UsdPhysics.PrismaticJoint):
        kind, typed = "prismatic", UsdPhysics.PrismaticJoint(prim)
    elif prim.IsA(UsdPhysics.RevoluteJoint):
        kind, typed = "revolute", UsdPhysics.RevoluteJoint(prim)
    else:
        raise ValueError(
            f"Articulated joint {prim.GetPath()} is a {prim.GetTypeName()}; only prismatic, revolute and fixed"
            " joints are supported."
        )
    return Joint(
        path=str(prim.GetPath()),
        kind=kind,
        axis=AXES.index(typed.GetAxisAttr().Get()),
        body0=str(typed.GetBody0Rel().GetTargets()[0]),
        body1=str(typed.GetBody1Rel().GetTargets()[0]),
        frame0=_frame(typed.GetLocalPos0Attr().Get(), typed.GetLocalRot0Attr().Get()),
        frame1=_frame(typed.GetLocalPos1Attr().Get(), typed.GetLocalRot1Attr().Get()),
    )


def _frame(position, rotation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([*rotation.GetImaginary(), rotation.GetReal()]).as_matrix()
    matrix[:3, 3] = list(position)
    return matrix
