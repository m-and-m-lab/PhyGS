# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot descriptions: every robot fact the framework reads, from one ``robot.yaml`` per robot.

No Isaac imports. A robot is one folder: its ``robot.yaml`` (the schema below), the files it names
(policy, calibrations, the measured transit footprint) and, only when the robot needs them, Python
plugins named ``file.py:attribute`` (a file in the folder) or ``package.module:attribute``.
:func:`load_robot` reads a robot of this repository by name (``robots/<name>/robot.yaml``) or one
anywhere else by path. Framework code asks the description for joints, links, tools and cameras by
their role; it never names a robot's parts itself. How to add a robot: ``docs/adding_a_robot.md``.

Frames: the framework's ``body`` frame is the robot's :attr:`RobotDescription.base_link`. Values of
gripper and arm joints are [m or rad, depending on joint type].
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import importlib.util
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROBOTS = Path(__file__).resolve().parent
SCHEMA = 1
KINEMATICS = ("holonomic", "differential")
# Camera roles: navigation (Nav2's point clouds), planning (the arm planner's depth), look (the camera
# look_at aims; at most one), front (the Front view of the UI and the video: one camera, or a left and
# a right one stitched).
ROLES = ("navigation", "planning", "look", "front")
# Gripper tool frames (gripper.tools): poses in the wrist link. "wrist" is the wrist link itself; every
# other tool has the gripper's axes: x the approach (out of the palm, toward the object), z the axis
# the jaws close along, y = z x x. Required: "hand" (the middle of the jaw gap) and "grasp" (where an
# object's grasp point goes). Optional, for the robot tests' scripted routines: "handle" (where a bar
# handle goes in a pull grasp) and "fixed_jaw" (a gripper with one fixed jaw: its inner pad, which a
# held object rests against; without it the jaws centre the object).
REQUIRED_TOOLS = ("wrist", "hand", "grasp")
# arm.motion (ManipulationController's MotionConfig): the limits and tolerances in joint units [m or
# rad] depend on the robot's joints and are required; the Cartesian tolerances and the waits may keep
# MotionConfig's defaults.
JOINT_MOTION = (
    "max_velocity",
    "max_acceleration",
    "tracking_tolerance",
    "joint_tolerance",
    "stopped_velocity",
    "gripper_tolerance",
    "gripper_stopped_velocity",
    "grip_squeeze",
)
OTHER_MOTION = (
    "tool_position_tolerance",
    "tool_orientation_tolerance",
    "position_tolerance",
    "orientation_tolerance",
    "settle_seconds",
    "timeout_seconds",
    "stop_timeout_seconds",
    "motion_window_seconds",
)


@dataclass(frozen=True)
class RobotInfo:
    """What an agent may read about its robot (``session.robot``): names, postures and gripper facts.

    ``stow`` and ``prestow`` [m or rad] are in :attr:`arm_joints` order; ``gripper_open`` and
    ``gripper_closed`` are the primary gripper joint's values; ``tools`` are the frames
    ``move_to_pose(tool=)`` accepts, each a pose in the wrist link [x, y, z [m], qx, qy, qz, qw] with
    the gripper's axes (x the approach, z the jaws' closing axis; ``wrist`` is the wrist link
    itself); ``look_camera`` is the camera ``look_at`` aims (None without).
    """

    name: str
    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]
    gripper_open: float
    gripper_closed: float
    tools: dict[str, tuple[float, ...]]
    cameras: tuple[str, ...]
    look_camera: str | None
    base_kinematics: str
    stow: tuple[float, ...]
    prestow: tuple[float, ...]
    move_seconds: float


@dataclass(frozen=True)
class RobotDescription:
    """One robot, validated at load; see the module docstring and ``robots/spot/robot.yaml``."""

    @dataclass(frozen=True)
    class Articulation:
        """PhysX articulation settings of the robot asset."""

        self_collisions: bool
        solver_position_iterations: int
        solver_velocity_iterations: int
        contact_sensors: bool

    @dataclass(frozen=True)
    class Base:
        """How the base moves. ``kind`` names the base controller (``legged_policy``, ``planar_joints``
        or a plugin ``module:Class``); ``kinematics`` is ``holonomic`` (v_x, v_y, v_rot) or
        ``differential`` (no v_y). ``config`` is the controller's settings and calibration file;
        ``options`` the rest of the ``base`` section, for the controller."""

        kind: str
        kinematics: str
        config: Path | None
        options: dict[str, Any]

    @dataclass(frozen=True)
    class Arm:
        """The arm: ``joints`` in planning order (the chain from the base link to ``wrist_link``),
        ``root_link`` (the arm's first link), ``stow`` (navigation's transit posture) and ``prestow``
        (just out of the stow, where planners start) [m or rad], ``move_seconds`` [s] (a skill's
        straight joint move) and ``motion`` (``MotionConfig`` settings)."""

        joints: tuple[str, ...]
        root_link: str
        wrist_link: str
        stow: tuple[float, ...]
        prestow: tuple[float, ...]
        move_seconds: float
        motion: dict[str, float]

        @property
        def transit_posture(self) -> tuple[tuple[str, float], ...]:
            """The stow as (joint, position [m or rad]) pairs: navigation's ``arm_posture``."""
            return tuple(zip(self.joints, self.stow, strict=True))

    @dataclass(frozen=True)
    class Gripper:
        """The gripper: ``joints`` with the primary first; each follows the primary times its
        ``ratios`` entry. ``open`` and ``closed`` are the primary's values [m or rad]. ``jaw_links``
        squeeze a grasped body. ``tools`` maps a tool name to its pose in the wrist link (4x4) [m], in
        the gripper's axes (``REQUIRED_TOOLS`` and the comment above it)."""

        joints: tuple[str, ...]
        ratios: tuple[float, ...]
        open: float
        closed: float
        jaw_links: tuple[str, ...]
        tools: dict[str, np.ndarray]

        @property
        def primary(self) -> str:
            return self.joints[0]

        def positions(self, value: float) -> dict[str, float]:
            """Every gripper joint's position for the primary at ``value`` [m or rad]."""
            return {joint: value * ratio for joint, ratio in zip(self.joints, self.ratios, strict=True)}

    @dataclass(frozen=True)
    class Camera:
        """A depth camera: ``prim`` below the robot prim (an existing camera, or where ``pinhole``
        spawns one at ``offset``), image ``width`` and ``height`` [px]."""

        prim: str
        width: int
        height: int
        pinhole: dict[str, Any] | None = None
        offset: dict[str, Any] | None = None

    @dataclass(frozen=True)
    class Navigation:
        """Nav2's limits (``NavigationConfig`` settings: period [s], max_speed [m/s], max_yaw_speed
        [rad/s]) and the measured transit footprint profile (rung 03, ``check_footprint.py``; None
        until it is measured: the robot cannot navigate yet)."""

        limits: dict[str, float]
        footprint_profile: Path | None

    @dataclass(frozen=True)
    class Planning:
        """The arm planners' robot model. ``default_joints`` [m or rad]: cuRobo's retract posture.
        ``usd_only_geometry``: name patterns of USD parts the URDF lacks (camera housings), boxed
        into the planning model. ``link_aliases``: USD body name regex -> URDF link (a foot body
        rigidly on its lower leg). ``surface_fit_links``: thin links cuRobo covers with small surface
        spheres. ``drawn_link``: the link whose path a plan's picture shows. ``workspace_lower`` and
        ``workspace_upper`` [m]: the depth map's corners in the body frame; ``obstacle_radius``
        [m]: known boxes within this of the base are the planner's obstacles (the arm's reach)."""

        default_joints: tuple[float, ...]
        usd_only_geometry: tuple[str, ...]
        link_aliases: dict[str, str]
        surface_fit_links: tuple[str, ...]
        drawn_link: str
        workspace_lower: tuple[float, float, float]
        workspace_upper: tuple[float, float, float]
        obstacle_radius: float

        def alias(self, body: str) -> str:
            """The URDF link a USD body belongs to."""
            for pattern, link in self.link_aliases.items():
                if re.fullmatch(pattern, body):
                    return re.sub(pattern, link, body)
            return body

        def usd_only(self, path: str) -> str | None:
            """The USD-only part a prim path is in (its first matching path element), or None."""
            return next(
                (part for part in path.split("/") if any(fnmatch.fnmatch(part, p) for p in self.usd_only_geometry)),
                None,
            )

    name: str
    folder: Path
    usd: Path
    urdf: Path
    base_link: str
    imu_link: str
    spawn_height: float
    fallen_height: float | None
    initial_joints: dict[str, float]
    articulation: Articulation
    actuators: str
    base: Base
    arm: Arm
    gripper: Gripper
    cameras: dict[str, Camera]
    camera_roles: dict[str, tuple[str, ...]]
    navigation: Navigation
    planning: Planning
    sha256: str

    @classmethod
    def load(cls, path: Path) -> RobotDescription:
        """Read and check a ``robot.yaml``: types, ranges, names it refers to and files it names."""
        path = Path(path).resolve(strict=True)
        text = path.read_text()
        data = yaml.safe_load(text)
        folder = path.parent
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path}: robot schema {data.get('schema')!r}; this code reads schema {SCHEMA}.")
        _keys(
            data,
            "the robot",
            required=("schema", "name", "usd", "urdf", "base_link", "spawn_height", "fallen_height", "initial_joints")
            + ("articulation", "actuators", "base", "arm", "gripper", "cameras", "camera_roles", "navigation")
            + ("planning",),
            optional=("imu_link", "fetch"),
        )

        def file(value: str, what: str) -> Path:
            target = (folder / value).resolve()
            if not target.is_file():
                hint = f" Fetch the robot's assets first: python3 {folder / data['fetch']}" if "fetch" in data else ""
                raise FileNotFoundError(f"{path}: {what} {target} does not exist.{hint}")
            return target

        base = dict(data["base"])
        _keys(base, "base", required=("kind", "kinematics"), optional=None)
        if base["kinematics"] not in KINEMATICS:
            raise ValueError(f"{path}: base kinematics {base['kinematics']!r}; expected one of {KINEMATICS}.")
        config = base.pop("config", None)
        arm = data["arm"]
        _keys(arm, "arm", required=("joints", "root_link", "wrist_link", "stow", "prestow", "move_seconds", "motion"))
        _keys(arm["motion"], "arm.motion", required=JOINT_MOTION, optional=OTHER_MOTION)
        gripper = data["gripper"]
        _keys(gripper, "gripper", required=("joints", "open", "closed", "jaw_links", "tools"))
        if not gripper["joints"] or list(gripper["joints"].values())[0] != 1.0:
            raise ValueError(f"{path}: gripper joints map joint -> ratio, the primary (first) at 1.0.")
        tools = {name: _pose(value, f"tool {name}") for name, value in gripper["tools"].items()}
        missing = [name for name in REQUIRED_TOOLS if name not in tools]
        if missing or not np.allclose(tools["wrist"], np.eye(4)):
            raise ValueError(f"{path}: tools need {REQUIRED_TOOLS}; 'wrist' is the wrist link itself.")
        if not all(
            np.allclose(pose[:3, :3], tools["hand"][:3, :3], atol=1e-6)
            for pose in tools.values()
            if pose is not tools["wrist"]
        ):
            raise ValueError(f"{path}: every tool but 'wrist' has the gripper's axes: one rotation, the hand's.")
        cameras = {}
        for name, camera in data["cameras"].items():
            _keys(camera, f"camera {name}", required=("prim", "width", "height"), optional=("pinhole", "offset"))
            cameras[name] = cls.Camera(
                camera["prim"], int(camera["width"]), int(camera["height"]), camera.get("pinhole"), camera.get("offset")
            )
        roles = {
            role: tuple(names if isinstance(names, list) else [names]) for role, names in data["camera_roles"].items()
        }
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f"{path}: unknown camera roles {sorted(unknown)}; expected {ROLES}.")
        for role, names in roles.items():
            if set(names) - set(cameras):
                raise ValueError(
                    f"{path}: camera role {role!r} names unknown cameras {sorted(set(names) - set(cameras))}."
                )
        if len(roles.get("look", ())) > 1 or not 1 <= len(roles.get("front", ("one",))) <= 2:
            raise ValueError(f"{path}: 'look' is one camera; 'front' one camera or a left and a right one.")
        navigation = data["navigation"]
        _keys(navigation, "navigation", required=("limits", "footprint_profile"))
        planning = data["planning"]
        _keys(
            planning,
            "planning",
            required=("default_joints", "usd_only_geometry", "link_aliases", "surface_fit_links", "drawn_link")
            + ("workspace", "obstacle_radius"),
        )
        workspace = planning["workspace"]
        articulation = data["articulation"]
        _keys(
            articulation,
            "articulation",
            required=("self_collisions", "solver_position_iterations", "solver_velocity_iterations", "contact_sensors"),
        )
        robot = cls(
            name=str(data["name"]),
            folder=folder,
            usd=file(data["usd"], "usd"),
            urdf=file(data["urdf"], "urdf"),
            base_link=data["base_link"],
            imu_link=data.get("imu_link", data["base_link"]),
            spawn_height=float(data["spawn_height"]),
            fallen_height=None if data["fallen_height"] is None else float(data["fallen_height"]),
            initial_joints={str(k): float(v) for k, v in data["initial_joints"].items()},
            articulation=cls.Articulation(**articulation),
            actuators=str(data["actuators"]),
            base=cls.Base(
                base.pop("kind"), base.pop("kinematics"), None if config is None else file(config, "base config"), base
            ),
            arm=cls.Arm(
                tuple(arm["joints"]),
                arm["root_link"],
                arm["wrist_link"],
                tuple(map(float, arm["stow"])),
                tuple(map(float, arm["prestow"])),
                float(arm["move_seconds"]),
                dict(arm["motion"]),
            ),
            gripper=cls.Gripper(
                tuple(gripper["joints"]),
                tuple(map(float, gripper["joints"].values())),
                float(gripper["open"]),
                float(gripper["closed"]),
                tuple(gripper["jaw_links"]),
                tools,
            ),
            cameras=cameras,
            camera_roles=roles,
            navigation=cls.Navigation(
                dict(navigation["limits"]),
                None if navigation["footprint_profile"] is None else file(navigation["footprint_profile"], "footprint"),
            ),
            planning=cls.Planning(
                tuple(map(float, planning["default_joints"])),
                tuple(planning["usd_only_geometry"]),
                dict(planning["link_aliases"]),
                tuple(planning["surface_fit_links"]),
                planning["drawn_link"],
                tuple(map(float, workspace["lower"])),
                tuple(map(float, workspace["upper"])),
                float(planning["obstacle_radius"]),
            ),
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        robot._check()
        return robot

    def _check(self) -> None:
        """Checks that need no URDF: sizes, finiteness, joint sets that must not overlap."""
        arm, gripper = self.arm, self.gripper
        values = (
            *arm.stow,
            *arm.prestow,
            arm.move_seconds,
            gripper.open,
            gripper.closed,
            *self.planning.default_joints,
        )
        if not all(map(math.isfinite, values)) or arm.move_seconds <= 0 or self.spawn_height < 0:
            raise ValueError(f"{self.name}: postures, gripper values and durations must be finite.")
        for name, posture in (
            ("stow", arm.stow),
            ("prestow", arm.prestow),
            ("default_joints", self.planning.default_joints),
        ):
            if len(posture) != len(arm.joints):
                raise ValueError(f"{self.name}: {name} needs one value per arm joint ({len(arm.joints)}).")
        if len(set(arm.joints) | set(gripper.joints)) != len(arm.joints) + len(gripper.joints):
            raise ValueError(f"{self.name}: arm and gripper joints must be distinct and unique.")
        named = [key for key in self.initial_joints if any(re.fullmatch(key, j) for j in arm.joints)]
        if named:
            raise ValueError(
                f"{self.name}: initial_joints must not set arm joints {named}; the arm starts at arm.stow."
            )
        lower, upper = np.array(self.planning.workspace_lower), np.array(self.planning.workspace_upper)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(lower < upper)
            or self.planning.obstacle_radius <= 0
        ):
            raise ValueError(f"{self.name}: the planning workspace needs lower < upper corners and a positive radius.")

    # Lookups.
    def cameras_for(self, role: str) -> tuple[str, ...]:
        """The cameras with ``role`` (:data:`ROLES`), in their configured order (none: empty)."""
        if role not in ROLES:
            raise ValueError(f"Unknown camera role {role!r}; expected one of {ROLES}.")
        return self.camera_roles.get(role, ())

    @property
    def look_camera(self) -> str | None:
        """The camera ``look_at`` aims, if the robot has one."""
        return next(iter(self.cameras_for("look")), None)

    @cached_property
    def arm_links(self) -> tuple[str, ...]:
        """URDF links the arm and gripper joints move, from the arm's root link outward."""
        children: dict[str, list[tuple[str, str]]] = {}
        for joint in self._urdf.findall("joint"):
            children.setdefault(joint.find("parent").get("link"), []).append(
                (joint.get("name"), joint.find("child").get("link"))
            )
        moving: list[str] = []
        pending = [self.arm.root_link]
        while pending:
            link = pending.pop(0)
            moving.append(link)
            pending.extend(child for _, child in children.get(link, ()))
        return tuple(moving)

    @cached_property
    def arm_joint_links(self) -> tuple[str, ...]:
        """The link each arm joint moves (its child), in joint order (Spot: arm_link_sh0 ... arm_link_wr1);
        for drawing the arm."""
        child = {joint.get("name"): joint.find("child").get("link") for joint in self._urdf.findall("joint")}
        return tuple(child[name] for name in self.arm.joints)

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """URDF limits [m or rad] of the movable joints."""
        limits = {}
        for joint in self._urdf.findall("joint"):
            limit = joint.find("limit")
            if joint.get("type") in {"revolute", "prismatic"} and limit is not None:
                limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
        return limits

    def tool(self, name: str) -> np.ndarray:
        """The tool frame ``name`` in the wrist link (4x4) [m]."""
        if name not in self.gripper.tools:
            raise ValueError(f"Unknown tool {name!r}; {self.name} has {sorted(self.gripper.tools)}.")
        return self.gripper.tools[name]

    def file(self, relative: str | Path) -> Path:
        """A file of the robot's folder."""
        return (self.folder / relative).resolve(strict=True)

    def plugin(self, reference: str) -> Any:
        """The object a ``file.py:attribute`` (in the robot's folder) or ``package.module:attribute``
        reference names."""
        if ":" not in reference:
            raise ValueError(f"{self.name}: plugin {reference!r} is not 'file.py:attribute' or 'module:attribute'.")
        source, attribute = reference.rsplit(":", 1)
        if source.endswith(".py"):
            path = self.file(source)
            module = _load_file(path)
        else:
            module = importlib.import_module(source)
        target = module
        for part in attribute.split("."):
            target = getattr(target, part)
        return target

    def info(self) -> RobotInfo:
        """The agent's view of the robot."""
        return RobotInfo(
            name=self.name,
            arm_joints=self.arm.joints,
            gripper_joints=self.gripper.joints,
            gripper_open=self.gripper.open,
            gripper_closed=self.gripper.closed,
            tools={name: _xyzw(pose) for name, pose in self.gripper.tools.items()},
            cameras=tuple(self.cameras),
            look_camera=self.look_camera,
            base_kinematics=self.base.kinematics,
            stow=self.arm.stow,
            prestow=self.arm.prestow,
            move_seconds=self.arm.move_seconds,
        )

    def summary(self) -> dict:
        """The robot for reports: its name, folder and the hash of its ``robot.yaml``."""
        return dict(name=self.name, folder=str(self.folder), robot_yaml_sha256=self.sha256)

    # The description check (rung "description" of the robot-test ladder): no simulator.
    def validate_urdf(self) -> None:
        """Check the description against its URDF: the base link is its root (the planners work in
        the root's frame), every named joint and link exists, the arm joints are the movable joints
        from the base link to the wrist link in order, and the postures and gripper values are within
        the joint limits. Raises ``ValueError`` listing every problem."""
        from phygs.execution.manipulation.planners.model import root_link

        urdf = self._urdf
        links = {link.get("name") for link in urdf.findall("link")}
        joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
        problems = []
        named_links = {
            "base_link": [self.base_link],
            "arm.root_link": [self.arm.root_link],
            "arm.wrist_link": [self.arm.wrist_link],
            "gripper.jaw_links": list(self.gripper.jaw_links),
            "planning.surface_fit_links": list(self.planning.surface_fit_links),
            "planning.drawn_link": [self.planning.drawn_link],
        }
        for what, names in named_links.items():
            problems += [f"{what}: no link {name!r}" for name in names if name not in links]
        try:
            if (root := root_link(urdf)) != self.base_link:
                problems.append(f"the root link is {root!r}, not base_link {self.base_link!r}")
        except ValueError as error:
            problems.append(str(error))
        for name in (*self.arm.joints, *self.gripper.joints):
            if name not in joints:
                problems.append(f"no joint {name!r}")
            elif joints[name].get("type") not in {"revolute", "prismatic"}:
                problems.append(
                    f"joint {name!r} is {joints[name].get('type')}; arm and gripper joints are revolute or prismatic"
                )
        if not problems:
            by_child = {joint.find("child").get("link"): joint for joint in joints.values()}
            chain, link = [], self.arm.wrist_link
            while link != self.base_link:
                if link not in by_child:
                    problems.append(f"no joint chain from {self.base_link!r} to {self.arm.wrist_link!r}")
                    break
                joint = by_child[link]
                if joint.get("type") != "fixed":
                    chain.append(joint.get("name"))
                link = joint.find("parent").get("link")
            else:
                if tuple(reversed(chain)) != self.arm.joints:
                    problems.append(
                        f"arm.joints {list(self.arm.joints)} are not the movable joints from {self.base_link!r} to"
                        f" {self.arm.wrist_link!r}: {list(reversed(chain))}"
                    )
            limits = self.joint_limits
            for what, posture in (("stow", self.arm.stow), ("prestow", self.arm.prestow)):
                for joint, value in zip(self.arm.joints, posture):
                    lower, upper = limits[joint]
                    if not lower <= value <= upper:
                        problems.append(f"arm.{what}: {joint}={value} outside [{lower}, {upper}]")
            for what, value in (("open", self.gripper.open), ("closed", self.gripper.closed)):
                for joint, position in self.gripper.positions(value).items():
                    lower, upper = limits[joint]
                    if not lower - 1e-9 <= position <= upper + 1e-9:
                        problems.append(f"gripper.{what}: {joint}={position} outside [{lower}, {upper}]")
        if problems:
            raise ValueError(f"{self.name}: the description does not match {self.urdf}:\n- " + "\n- ".join(problems))

    def validate_usd(self) -> None:
        """Check the description against its USD (``pxr``; no simulator): the arm, gripper and base
        joints are joints of the asset, the named links are its rigid bodies, every camera prim (a
        spawned camera's parent) exists, and every ``initial_joints`` pattern matches a joint. Raises
        ``ValueError`` listing every problem."""
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(self.usd))
        root = stage.GetDefaultPrim()
        prims = list(Usd.PrimRange(root))
        joints = {p.GetName() for p in prims if p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint)}
        bodies = {p.GetName() for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        base_joints = tuple(self.base.options.get("joints", {}).values())
        problems = [
            f"no joint {name!r}"
            for name in (*self.arm.joints, *self.gripper.joints, *base_joints)
            if name not in joints
        ]
        links = (self.base_link, self.imu_link, self.arm.wrist_link, *self.gripper.jaw_links)
        problems += [f"no rigid body {name!r}" for name in dict.fromkeys(links) if name not in bodies]
        for name, camera in self.cameras.items():
            path = root.GetPath().AppendPath(camera.prim)
            if not stage.GetPrimAtPath(path.GetParentPath() if camera.pinhole else path).IsValid():
                problems.append(f"camera {name!r}: no prim {path}" + (" parent" if camera.pinhole else ""))
        problems += [
            f"initial_joints {key!r} matches no joint"
            for key in self.initial_joints
            if not any(re.fullmatch(key, j) for j in joints)
        ]
        if problems:
            raise ValueError(f"{self.name}: the description does not match {self.usd}:\n- " + "\n- ".join(problems))

    @cached_property
    def _urdf(self) -> ET.Element:
        return ET.parse(self.urdf).getroot()


def load_robot(robot: str | Path) -> RobotDescription:
    """A robot by name (``robots/<name>/robot.yaml`` of this repository), or by the path of its
    ``robot.yaml`` or of its folder."""
    path = Path(robot)
    if not path.suffix and not path.is_absolute() and len(path.parts) == 1 and not path.exists():
        path = ROBOTS / str(robot)
    if path.is_dir():
        path = path / "robot.yaml"
    if not path.is_file():
        known = sorted(p.parent.name for p in ROBOTS.glob("[!_]*/robot.yaml"))
        raise FileNotFoundError(f"No robot {robot!r}: not a robot.yaml or folder, and not one of {known}.")
    return RobotDescription.load(path)


def _keys(section: dict, what: str, *, required: tuple[str, ...], optional: tuple[str, ...] | None = ()) -> None:
    """Fail on missing keys, and on unknown ones unless ``optional`` is None (any extra key allowed)."""
    if not isinstance(section, dict):
        raise ValueError(f"{what} must be a mapping.")
    missing = [key for key in required if key not in section]
    unknown = [] if optional is None else sorted(set(section) - set(required) - set(optional))
    if missing or unknown:
        raise ValueError(f"{what}: missing keys {missing}, unknown keys {unknown}.")


def _pose(value: dict, what: str) -> np.ndarray:
    """A 4x4 transform [m] from ``{position: [x, y, z], rotation_xyzw: [qx, qy, qz, qw]}`` (rotation optional)."""
    from scipy.spatial.transform import Rotation

    _keys(value, what, required=("position",), optional=("rotation_xyzw",))
    matrix = np.eye(4)
    matrix[:3, 3] = [float(v) for v in value["position"]]
    if "rotation_xyzw" in value:
        matrix[:3, :3] = Rotation.from_quat([float(v) for v in value["rotation_xyzw"]]).as_matrix()
    matrix.flags.writeable = False
    return matrix


def _xyzw(pose: np.ndarray) -> tuple[float, ...]:
    """A 4x4 transform as [x, y, z [m], qx, qy, qz, qw]."""
    from scipy.spatial.transform import Rotation

    return (*map(float, pose[:3, 3]), *map(float, Rotation.from_matrix(pose[:3, :3]).as_quat()))


def _load_file(path: Path):
    """Import a robot folder's Python file once per path (two folders may share a robot name)."""
    import sys

    name = f"phygs_robot_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module
