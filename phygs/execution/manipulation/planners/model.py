# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The arm planners' robot model: the robot's URDF meshes plus boxes around USD-only parts (camera
housings); USD spheres for the referee and MoveIt's arm. Its frame is the URDF's root link, the
robot's base link (the framework's ``body`` frame)."""

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


def prepare_model(robot, output: Path, *, geometry: dict, locked_joints: dict, usd_boxes: dict) -> dict:
    """Write the planning URDF and SRDF of ``robot`` (a robot description), every joint that is not
    the arm's locked at its measured position in ``locked_joints`` [m or rad].

    The URDF has the collision meshes plus boxes around the USD geometry the URDF lacks
    (``usd_boxes``, from :func:`usd_only_boxes`); cuRobo fits its own spheres to them.
    ``geometry`` holds the conservative USD spheres the independent referee checks the arm
    with. MoveIt's URDF (``moveit_urdf``) gives the arm's links those spheres instead of their
    meshes: they reach up to about 5 cm past the meshes, so a path planned around the meshes
    could still fail the referee. The SRDF (MoveIt only) skips pairs over MoveIt's links, and
    pairs of sphere-modelled links two joints apart: their spheres overlap around the joint
    between them (wrist and forearm after a drawer release) where the meshes do not touch.

    The returned model is what the planner worker reads: the files, the arm and gripper joints, the
    links the planners treat specially, and the robot's planning settings.
    """
    from phygs.execution.manipulation.controller import MotionConfig

    source = robot.urdf
    joints = tuple(robot.arm.joints)
    tree = ET.parse(source)
    root = tree.getroot()
    if not geometry or not locked_joints:
        raise ValueError("Full geometry and measured non-arm joint positions are required")
    movable = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    missing = sorted(set(movable) - set(joints) - set(locked_joints))
    if missing:
        raise ValueError(f"Measured positions are missing for the model's joints {missing}")
    locked_joints = {name: float(locked_joints[name]) for name in movable if name not in joints}
    links = {link.get("name"): link for link in root.findall("link")}
    if not set(usd_boxes) <= set(links):
        raise ValueError(f"USD geometry on unknown links: {sorted(set(usd_boxes) - set(links))}")
    for name, link in links.items():
        for mesh in link.iter("mesh"):  # Visual meshes stay: cuRobo's builder fits only links that have one.
            mesh.set("filename", (source.parent / mesh.get("filename")).resolve().as_uri())
        for box in usd_boxes.get(name, []):
            collision = ET.SubElement(link, "collision", name="usd_only")
            ET.SubElement(collision, "origin", xyz=" ".join(map(str, box["center"])))
            ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=" ".join(map(str, box["size"])))
    motion = MotionConfig(**robot.arm.motion)
    for joint in root.findall("joint"):
        if joint.get("name") in joints:
            joint.find("limit").set("velocity", str(motion.max_velocity))
    colliding = [name for name, link in links.items() if link.find("collision") is not None]
    ignored = self_collision_ignore(root, colliding, joints, robot.gripper.jaw_links)
    output.mkdir(parents=True, exist_ok=True)
    urdf = output / "planning.urdf"
    tree.write(urdf)
    arm_links = set(robot.arm_links)
    moveit_tree = copy.deepcopy(tree)
    for link in moveit_tree.getroot().findall("link"):
        spheres = geometry.get(link.get("name")) if link.get("name") in arm_links else None
        if not spheres:
            continue
        for collision in link.findall("collision"):
            link.remove(collision)
        for sphere in spheres:
            collision = ET.SubElement(link, "collision", name="referee_sphere")
            ET.SubElement(collision, "origin", xyz=" ".join(map(str, sphere["center"])))
            ET.SubElement(ET.SubElement(collision, "geometry"), "sphere", radius=str(sphere["radius"]))
    moveit_urdf = output / "planning_moveit.urdf"
    moveit_tree.write(moveit_urdf)
    moveit_root = moveit_tree.getroot()
    moveit_colliding = [link.get("name") for link in moveit_root.findall("link") if link.find("collision") is not None]
    semantic = ET.Element("robot", name=robot.name)
    chain = dict(base_link=robot.base_link, tip_link=robot.arm.wrist_link)
    ET.SubElement(ET.SubElement(semantic, "group", name="arm"), "chain", **chain)
    for a, others in self_collision_ignore(moveit_root, moveit_colliding, joints, robot.gripper.jaw_links).items():
        for b in others:
            if a < b:
                ET.SubElement(semantic, "disable_collisions", link1=a, link2=b, reason="Adjacent or both static")
    sphered = [name for name in moveit_colliding if name in arm_links and geometry.get(name)]
    for a, b in joint_neighbours(moveit_root, sphered, hops=2):
        ET.SubElement(semantic, "disable_collisions", link1=a, link2=b, reason="Spheres overlap at the joint")
    srdf = output / "planning.srdf"
    ET.ElementTree(semantic).write(srdf)
    gripper = robot.gripper
    return dict(
        name=robot.name,
        urdf=str(urdf.resolve()),
        moveit_urdf=str(moveit_urdf.resolve()),
        srdf=str(srdf.resolve()),
        base_link=robot.base_link,
        wrist_link=robot.arm.wrist_link,
        joints=list(joints),
        arm_links=list(robot.arm_links),
        # May touch a held object: the jaws and the palm (the wrist link; a two-finger hand holds an
        # object's top close under it).
        held_contact_links=list(dict.fromkeys((robot.arm.wrist_link, *gripper.jaw_links))),
        # Locked in the model at the primary's position [m or rad]; a request may give its measured one.
        gripper_joint=gripper.primary,
        gripper_ratios=dict(zip(gripper.joints, gripper.ratios, strict=True)),
        surface_fit_links=list(robot.planning.surface_fit_links),
        default_joints=list(robot.planning.default_joints),
        max_velocity=motion.max_velocity,  # The arm joints' limits [m/s or rad/s, /s²].
        max_acceleration=motion.max_acceleration,
        spheres=geometry,
        self_collision_ignore=ignored,
        locked_joints=locked_joints,
    )


def gripper_positions(model: dict, value: float) -> dict[str, float]:
    """Every gripper joint's position [m or rad] with the model's primary gripper joint at ``value``."""
    return {joint: value * ratio for joint, ratio in model["gripper_ratios"].items()}


def planner_angle(gripper: float) -> float:
    """The gripper angle [rad] a cuRobo planner is built for: one planner per 0.01 rad, since cuRobo
    fixes locked joints when it builds a planner."""
    return round(gripper, 2)


def joint_neighbours(root: ET.Element, links: list[str], *, hops: int) -> list[tuple[str, str]]:
    """Pairs (a < b) of links exactly ``hops`` moving joints apart; links joined by fixed joints count as one."""
    groups = {link.get("name"): link.get("name") for link in root.findall("link")}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            parent, child = (joint.find(k).get("link") for k in ("parent", "child"))
            old, new = groups[child], groups[parent]
            groups = {k: new if v == old else v for k, v in groups.items()}
    edges: dict[str, set[str]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            a, b = (groups[joint.find(k).get("link")] for k in ("parent", "child"))
            edges.setdefault(a, set()).add(b)
            edges.setdefault(b, set()).add(a)

    def distance(a: str, b: str) -> int | None:
        frontier, seen = {a}, {a}
        for step in range(1, hops + 1):
            frontier = {n for f in frontier for n in edges.get(f, ())} - seen
            if b in frontier:
                return step
            seen |= frontier
        return None

    return [(a, b) for a in links for b in links if a < b and distance(groups[a], groups[b]) == hops]


def self_collision_ignore(
    root: ET.Element, links: list[str], arm_joints: tuple[str, ...], jaws: tuple[str, ...] = ()
) -> dict[str, list[str]]:
    """Link pairs never checked against each other: same rigid group, joined, both static, or both
    jaws (``jaws``: the gripper's jaw links, which close onto each other: two fingers of a closed
    gripper touch).

    Non-arm joints are locked during arm planning, so links the arm (``arm_joints``) does not move
    never move relative to each other.
    """
    groups = {link.get("name"): link.get("name") for link in root.findall("link")}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            parent, child = (joint.find(k).get("link") for k in ("parent", "child"))
            old, new = groups[child], groups[parent]
            groups = {k: new if v == old else v for k, v in groups.items()}
    adjacent = {
        frozenset((groups[j.find("parent").get("link")], groups[j.find("child").get("link")]))
        for j in root.findall("joint")
    }
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint.find("child").get("link"))
    moving = [j.find("child").get("link") for j in root.findall("joint") if j.get("name") in arm_joints]
    for link in moving:
        moving.extend(child for child in children.get(link, []) if child not in moving)
    static = {link for link in links if link not in moving}
    return {
        a: [
            b
            for b in links
            if b != a
            and (
                groups[a] == groups[b]
                or frozenset((groups[a], groups[b])) in adjacent
                or {a, b} <= static
                or {a, b} <= set(jaws)
            )
        ]
        for a in links
    }


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray | None
    prismatic: bool


def root_link(urdf: ET.Element) -> str:
    """The URDF's root link: the one no joint moves."""
    joints = urdf.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    links = {link.get("name") for link in urdf.findall("link")} | {j.find("parent").get("link") for j in joints}
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"A robot URDF has one root link; found {roots}")
    return roots[0]


@lru_cache(maxsize=8)
def _joint_chain(urdf: str, modified_ns: int, size: int, inode: int) -> tuple[str, tuple[_Joint, ...]]:
    """Load the root link and static joint geometry once per file version; keep no measured state."""
    from scipy.spatial.transform import Rotation

    tree = ET.parse(urdf).getroot()
    pending = list(tree.findall("joint"))
    root = root_link(tree)
    available = {root}
    chain = []
    while pending:
        advanced = []
        for joint in pending:
            parent, child = (joint.find(k).get("link") for k in ("parent", "child"))
            if parent not in available:
                continue
            origin = joint.find("origin")
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                matrix[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            axis = None
            if joint.get("type") in {"revolute", "continuous", "prismatic"}:
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                axis.flags.writeable = False
            elif joint.get("type") != "fixed":
                raise ValueError("The planning model supports fixed, revolute and prismatic joints only")
            matrix.flags.writeable = False
            chain.append(_Joint(joint.get("name"), parent, child, matrix, axis, joint.get("type") == "prismatic"))
            available.add(child)
            advanced.append(joint)
        if not advanced:
            raise ValueError("URDF contains disconnected links")
        pending = [j for j in pending if j not in advanced]
    return root, tuple(chain)


def link_transforms(urdf: str, joints: dict[str, float]) -> dict[str, np.ndarray]:
    """Compute link poses [m] in the URDF's root link (the body frame) from joint positions [m or rad].

    Static definitions are cached by resolved path and file modification metadata.
    Regenerated model files reload automatically; measured poses are never cached.
    """
    from scipy.spatial.transform import Rotation

    path = Path(urdf).resolve(strict=True)
    stat = path.stat()
    root, chain = _joint_chain(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
    transforms = {root: np.eye(4)}
    for joint in chain:
        matrix = joint.origin.copy()
        if joint.prismatic:
            matrix[:3, 3] += matrix[:3, :3] @ (joint.axis * joints[joint.name])
        elif joint.axis is not None:
            matrix[:3, :3] = matrix[:3, :3] @ Rotation.from_rotvec(joint.axis * joints[joint.name]).as_matrix()
        transforms[joint.child] = transforms[joint.parent] @ matrix
    return transforms


def transformed_spheres(model: dict, joints: dict[str, float], world_from_body: np.ndarray) -> np.ndarray:
    """Return collision sphere centers and radii [x,y,z,r] in world coordinates [m]."""
    frames = link_transforms(model["urdf"], joints)
    rows = []
    for name, spheres in model["spheres"].items():
        pose = world_from_body @ frames[name]
        centers = np.array([sphere["center"] for sphere in spheres])
        radii = np.array([sphere["radius"] for sphere in spheres])
        world_centers = centers @ pose[:3, :3].T + pose[:3, 3]
        rows.append(np.column_stack((world_centers, radii)))
    return np.concatenate(rows)


def audit_collision_geometry(stage, robot_path: str, model: dict, planning) -> dict:
    """Measure USD collider-vertex coverage by link-local planner spheres [m]; ``planning`` is the
    robot description's planning section (its USD-only parts and link aliases).

    This identifies missing geometry; covered vertices do not certify triangle
    interiors or swept motion. It never changes the model to conceal a mismatch.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(robot_path)
    if not root.IsValid():
        raise ValueError(f"Robot prim does not exist: {robot_path}")
    cache = UsdGeom.XformCache()
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    entries = []
    for prim in Usd.PrimRange(root):
        is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        is_camera_mesh = _usd_only_mesh(prim, planning)
        if not is_collider and not is_camera_mesh:
            continue
        if is_collider and not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        parent = prim
        while parent != root and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        name = planning.alias(parent.GetName())
        if name != parent.GetName():
            parent = stage.GetPrimAtPath(robot_path + "/" + name)
        spheres = model["spheres"].get(name)
        if not spheres:
            entries.append(dict(path=str(prim.GetPath()), link=name, status="missing_link_geometry"))
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            if points.size == 0:
                raise ValueError(f"Empty collision mesh: {prim.GetPath()}")
            transform = np.array(cache.ComputeRelativeTransform(prim, parent)[0]).T
            points = points @ transform[:3, :3].T + transform[:3, 3]
        else:
            bounds = bounds_cache.ComputeRelativeBound(prim, parent).ComputeAlignedRange()
            lower, upper = np.array(bounds.GetMin()), np.array(bounds.GetMax())
            points = np.array(
                [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
            )
        distances = np.full(len(points), np.inf)
        for sphere in spheres:
            distances = np.minimum(distances, np.linalg.norm(points - sphere["center"], axis=1) - sphere["radius"])
        entries.append(
            dict(
                path=str(prim.GetPath()),
                link=name,
                status="covered_vertices" if np.max(distances) <= 1e-6 else "uncovered_vertices",
                maximum_uncovered_distance_m=float(max(0.0, np.max(distances))),
                uncovered_vertices=int(np.count_nonzero(distances > 1e-6)),
                vertices=len(points),
            )
        )
    if not entries:
        raise ValueError("No robot collision geometry found")
    return dict(
        qualified=all(e["status"] == "covered_vertices" for e in entries),
        checks=entries,
        scope="Collider and camera-mesh vertices only; no swept-volume certificate",
    )


def _usd_only_mesh(prim, planning) -> bool:
    """Whether a prim is a mesh of a USD-only part (a camera housing; ``planning.usd_only_geometry``)."""
    from pxr import UsdGeom

    path = str(prim.GetPath())
    return (
        planning.usd_only(path) is not None and prim.IsA(UsdGeom.Mesh) and "OmniverseKitViewportCameraMesh" not in path
    )


def _link_geometry(stage, robot_path: str, planning):
    """Yield (link, link prim, prim, lower, upper, is_collider) for enabled colliders and USD-only meshes.

    Bounds [m] are axis-aligned in the link frame; a body with a link alias (a foot) is expressed in
    the aliased link's frame.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(robot_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    for prim in Usd.PrimRange(root):
        collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        camera = _usd_only_mesh(prim, planning)
        if not collider and not camera:
            continue
        if collider and not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        parent = prim
        while parent != root and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        name = planning.alias(parent.GetName())
        if name != parent.GetName():
            parent = stage.GetPrimAtPath(robot_path + "/" + name)
        if not parent.IsValid() or parent == root:
            raise ValueError(f"No modeled rigid link for {prim.GetPath()}")
        bounds = cache.ComputeRelativeBound(prim, parent).ComputeAlignedRange()
        lower, upper = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if not np.isfinite([lower, upper]).all() or np.any(upper < lower):
            raise ValueError(f"Invalid collider bounds: {prim.GetPath()}")
        yield name, parent, prim, lower, upper, collider


def usd_only_boxes(stage, robot_path: str, planning) -> dict[str, list[dict]]:
    """Boxes around USD geometry the URDF lacks: center and size [m] in the link frame.

    One box per USD-only part (``planning.usd_only_geometry``, e.g. a camera housing the URDF does
    not have) and one per collider outside the URDF importer's ``collisions`` scope, e.g. the grip
    spheres on Spot's jaw.
    """
    parts: dict[tuple[str, str], list[np.ndarray]] = {}
    for name, _, prim, lower, upper, collider in _link_geometry(stage, robot_path, planning):
        path = str(prim.GetPath())
        if collider and "collisions" in path.split("/"):
            continue  # Imported from the URDF (the importer's ``collisions`` scope, or a prim of that name).
        part = path if collider else planning.usd_only(path)
        parts.setdefault((name, part), []).extend((lower, upper))
    boxes: dict[str, list[dict]] = {}
    for (name, _), corners in sorted(parts.items()):
        lower, upper = np.min(corners, axis=0), np.max(corners, axis=0)
        boxes.setdefault(name, []).append(dict(center=((lower + upper) / 2).tolist(), size=(upper - lower).tolist()))
    return boxes


def collider_spheres(stage, robot_path: str, planning, *, cell_size: float = 0.04) -> dict:
    """Conservatively cover USD collider and USD-only part bounds with spheres [m].

    A regular grid covers each link-local AABB. For authored convex hulls,
    cells separated by a hull plane are excluded. Every retained cell is enclosed by its
    sphere, including interiors. This may reject tight passages; it cannot certify
    exact mesh clearance. Aliased bodies (feet) are expressed in their link's frame.
    """
    from pxr import UsdGeom, UsdPhysics

    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("Geometry cell size must be positive")
    result = {}
    for name, parent, prim, lower, upper, collider in _link_geometry(stage, robot_path, planning):
        count = np.maximum(1, np.ceil((upper - lower) / cell_size).astype(int))
        width = (upper - lower) / count
        radius = float(np.linalg.norm(width) / 2)
        if radius == 0:
            raise ValueError(f"Zero-volume collider: {prim.GetPath()}")
        centers = np.stack(
            np.meshgrid(*(lower[i] + (np.arange(count[i]) + 0.5) * width[i] for i in range(3)), indexing="ij"), -1
        ).reshape(-1, 3)
        if (
            collider
            and prim.IsA(UsdGeom.Mesh)
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
        ):
            from scipy.spatial import ConvexHull

            transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, parent)
            transform = np.array(transform).T
            vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            planes = ConvexHull(vertices).equations
            # A cell is outside if its minimum signed distance to any hull
            # plane is positive. Retain all other cells, including boundary cells.
            minimum = centers @ planes[:, :3].T + planes[:, 3] - (width / 2) @ np.abs(planes[:, :3]).T
            centers = centers[np.all(minimum <= 1e-10, axis=1)]
        result.setdefault(name, []).extend(dict(center=p.tolist(), radius=radius) for p in centers)
    if not result:
        raise ValueError("No robot geometry was extracted")
    return result
