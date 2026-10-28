# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged sliding-part (drawer) information from a scene's prismatic joints and body poses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from phygs.execution.manipulation.skills.types import ArticulatedPart

AXES = "XYZ"


def sliding_part(
    stage: Usd.Stage, joint_path: str, handle_path: str, body_pose: Callable[[str], np.ndarray]
) -> ArticulatedPart:
    """The part moved by the prismatic joint at joint_path, with the cylinder at handle_path as its handle.

    Joint frames, limits and the handle's place on the moving body come from the authored
    stage; ``body_pose(path)`` gives a rigid body's world pose now (4x4, m), from the
    simulation. A joint without body0 is attached to the world.
    """
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_path))
    handle = UsdGeom.Cylinder(stage.GetPrimAtPath(handle_path))
    if not joint or not handle:
        raise ValueError(f"Expected a prismatic joint at {joint_path} and a cylinder at {handle_path}.")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Sliding parts need a stage in meters.")
    frames = []
    for body, position, rotation in (
        (joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
        (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
    ):
        targets = body.GetTargets()
        world_from_body = body_pose(str(targets[0])) if targets else np.eye(4)
        frames.append(world_from_body @ transform(position.Get(), rotation.Get()))
    direction = frames[0][:3, AXES.index(joint.GetAxisAttr().Get())]
    moving = str(joint.GetBody1Rel().GetTargets()[0])
    cache = UsdGeom.XformCache()
    authored_body = usd_matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(moving)))
    authored_handle = usd_matrix(cache.GetLocalToWorldTransform(handle.GetPrim()))
    world_from_handle = body_pose(moving) @ np.linalg.inv(authored_body) @ authored_handle
    if not np.allclose(np.linalg.norm(world_from_handle[:3, :3], axis=0), 1.0, atol=1e-6):
        raise ValueError(f"Scaled handles are not supported: {handle_path}")
    return ArticulatedPart(
        handle_center=tuple(world_from_handle[:3, 3].tolist()),
        handle_axis=tuple(world_from_handle[:3, AXES.index(handle.GetAxisAttr().Get())].tolist()),
        handle_length=float(handle.GetHeightAttr().Get()),
        handle_radius=float(handle.GetRadiusAttr().Get()),
        direction=tuple(direction.tolist()),
        opening=float(np.dot(frames[1][:3, 3] - frames[0][:3, 3], direction)),
        limits=(float(joint.GetLowerLimitAttr().Get()), float(joint.GetUpperLimitAttr().Get())),
    )


def transform(position, rotation) -> np.ndarray:
    """4x4 transform from a USD position [m] and quaternion."""
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    imaginary = rotation.GetImaginary()
    matrix[:3, :3] = Rotation.from_quat([*imaginary, rotation.GetReal()]).as_matrix()
    matrix[:3, 3] = list(position)
    return matrix


def usd_matrix(matrix) -> np.ndarray:
    """Column-vector 4x4 from a USD (row-vector) matrix."""
    return np.array(matrix, dtype=float).T
