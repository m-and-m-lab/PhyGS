# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write ``planning.urdf``, the arm planners' model of the Ridgeback-Franka, from its USD.

The kinematics (joint frames, axes and limits) come from the USD's physics joints below the base
link, so the model matches the simulated robot. The Franka links get the collision meshes of
``franka_description`` (``meshes/``, Apache-2.0, ``meshes/LICENSE``); the base and its fixed parts
get boxes around their USD colliders. Run once after fetching the assets (``fetch.py``), with any
Python that has ``pxr`` (Isaac Sim's)::

    ./isaaclab.sh -p scripts/PhyGS/phygs/robots/ridgeback_franka/make_planning_urdf.py
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT_LINK = "base_link"  # The planning model's root: the robot's base link (above the planar joints).
MESHES = {  # Franka link -> collision mesh in meshes/ (franka_description), and its pose in the link.
    **{f"panda_link{i}": (f"link{i}.obj", (0.0, 0.0, 0.0)) for i in range(8)},
    "panda_hand": ("hand.obj", (0.0, 0.0, 0.0)),
    "panda_leftfinger": ("finger.obj", (0.0, 0.0, 0.0)),
    "panda_rightfinger": ("finger.obj", (0.0, 0.0, math.pi)),
}


def matrix(position, quaternion_wxyz) -> np.ndarray:
    pose = np.eye(4)
    w, x, y, z = quaternion_wxyz
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = position
    return pose


def joint_origin(joint: UsdPhysics.Joint) -> tuple[np.ndarray, np.ndarray | None]:
    """URDF origin (4x4) and axis of a USD joint whose child frame sits at the joint (localPos1 = 0):
    the child moves as parent @ T0 @ M(q) @ inv(R1) = parent @ (T0 inv(R1)) @ M'(q), M' about R1 x."""
    prim = joint.GetPrim()
    position1 = np.array(joint.GetLocalPos1Attr().Get())
    if np.linalg.norm(position1) > 1e-9:
        raise ValueError(f"{prim.GetPath()}: the child frame is off the joint ({position1}).")
    q0, q1 = joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()
    t0 = matrix(joint.GetLocalPos0Attr().Get(), (q0.GetReal(), *q0.GetImaginary()))
    r1 = matrix((0.0, 0.0, 0.0), (q1.GetReal(), *q1.GetImaginary()))
    axis = None
    if not prim.IsA(UsdPhysics.FixedJoint):
        unit = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}[
            prim.GetAttribute("physics:axis").Get()
        ]
        axis = r1[:3, :3] @ np.array(unit)
    return t0 @ np.linalg.inv(r1), axis


def collider_box(stage: Usd.Stage, body: Usd.Prim) -> tuple[np.ndarray, np.ndarray] | None:
    """Center and size [m] of the box around a body's colliders, in the body's frame."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    corners = []
    for prim in Usd.PrimRange(body):
        if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            break
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bounds = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
            corners += [np.array(bounds.GetMin()), np.array(bounds.GetMax())]
    if not corners:
        return None
    lower, upper = np.min(corners, axis=0), np.max(corners, axis=0)
    return (lower + upper) / 2, upper - lower


def element(parent: ET.Element, tag: str, **attributes) -> ET.Element:
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attributes.items()})


def vector(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def main() -> None:
    stage = Usd.Stage.Open(str(HERE / "assets" / "ridgeback_franka.usd"))
    root = stage.GetDefaultPrim()
    joints = [UsdPhysics.Joint(prim) for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)]
    by_parent: dict[str, list[UsdPhysics.Joint]] = {}
    for joint in joints:
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if parents and children:
            by_parent.setdefault(parents[0].name, []).append(joint)
    robot = ET.Element("robot", name="ridgeback_franka")
    pending = [ROOT_LINK]
    while pending:
        link = pending.pop(0)
        node = element(robot, "link", name=link)
        if link in MESHES:
            mesh, rpy = MESHES[link]
            for kind in ("visual", "collision"):  # cuRobo fits spheres to links with a visual mesh.
                geometry = element(node, kind)
                element(geometry, "origin", xyz="0 0 0", rpy=vector(rpy))
                element(element(geometry, "geometry"), "mesh", filename=f"meshes/{mesh}")
        else:
            box = collider_box(stage, root.GetChild(link))
            if box is not None:
                for kind in ("visual", "collision"):
                    geometry = element(node, kind)
                    element(geometry, "origin", xyz=vector(box[0]), rpy="0 0 0")
                    element(element(geometry, "geometry"), "box", size=vector(box[1]))
        for joint in by_parent.get(link, []):
            prim, child = joint.GetPrim(), joint.GetBody1Rel().GetTargets()[0].name
            origin, axis = joint_origin(joint)
            kind = "fixed" if axis is None else "prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else "revolute"
            node = element(robot, "joint", name=prim.GetName(), type=kind)
            element(node, "parent", link=link)
            element(node, "child", link=child)
            rpy = Rotation.from_matrix(origin[:3, :3]).as_euler("xyz")
            element(node, "origin", xyz=vector(origin[:3, 3]), rpy=vector(rpy))
            if axis is not None:
                element(node, "axis", xyz=vector(axis))
                lower, upper = (prim.GetAttribute(f"physics:{end}Limit").Get() for end in ("lower", "upper"))
                scale = 1.0 if kind == "prismatic" else math.pi / 180.0  # USD revolute limits are degrees.
                element(
                    node, "limit", lower=f"{lower * scale:.6f}", upper=f"{upper * scale:.6f}", effort=100, velocity=2.0
                )
            pending.append(child)
    robot.insert(0, ET.Comment(" Generated by make_planning_urdf.py from assets/ridgeback_franka.usd; do not edit. "))
    ET.indent(robot)
    (HERE / "planning.urdf").write_text(ET.tostring(robot, encoding="unicode") + "\n")
    print(f"wrote {HERE / 'planning.urdf'}")


if __name__ == "__main__":
    main()
