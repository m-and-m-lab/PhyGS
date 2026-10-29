# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What the agent-shaped robot tests share: an episode spec run with a scripted plan as the agent.

A test's plan drives the robot through the agent session only, as any agent does, and knows only
what the spec grants (``session.hints``). The test measures on the other side: ``world.truth()``
after the run, and during it the per-tick call ``World.run(during=...)``, which may also move
a scene body (:func:`move_body`). The plan names its steps (:class:`Steps`) so the test knows what
it measures. Pass thresholds are in each test's ``configs/*.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # phygs from the checkout.

from phygs_robot_tests import routines  # noqa: E402
from phygs_robot_tests.common import CONFIGS, Check, parser, robot  # noqa: E402

SETTINGS = yaml.safe_load((CONFIGS / "episodes.yaml").read_text())


def start(description: str, default_spec: str, extra=None):
    """Parse ``--spec``, ``--clock`` and the shared flags (``--record`` among them), plus the test's own
    (``extra(flags)`` adds them); start the app. Returns (check, arguments, spec)."""
    from phygs.world.spec import EpisodeSpec

    flags = parser(description)
    flags.add_argument("--spec", type=Path, default=CONFIGS / "episodes" / f"{default_spec}.yaml")
    flags.add_argument("--clock", choices=("charged", "realtime"), default="charged")
    flags.set_defaults(enable_cameras=True)
    if extra is not None:
        extra(flags)
    args = flags.parse_args()
    spec = EpisodeSpec.load(args.spec)
    args.robot = robot(args, default=spec.robot.model)  # The spec's robot, or --robot.
    spec = spec.with_robot(args.robot.name)  # Its start for this robot, as the World uses it.
    routines.use_scene(spec.scene.name)  # The plans' stances for this scene.
    routines.use_robot(args.robot.info())  # The plans' gripper geometry and arm postures for this robot.
    return Check(f"{Path(sys.argv[0]).stem} {spec.id}", args), args, spec


def world(args, spec, planner: str | None = None):
    """The episode's World (a context manager); ``planner`` (``curobo`` or ``moveit``) overrides the
    arm planner backend of ``phygs/configs/environment.yaml``."""
    from phygs_robot_tests.tasks import load_task

    from phygs.world import World

    return World(
        spec,
        args.output,
        clock=args.clock,
        record=args.record,
        device=args.device,
        overrides=None if planner is None else {"planner": {"backend": planner}},
        robot=args.robot,
        task=load_task(spec.task),
    )


class Steps:
    """The plan's named steps, shared with the test's per-tick call (``World.run(during=steps)``).

    The plan calls :meth:`enter`; :attr:`current` is the step running. At the first tick after a step
    was entered the test's ``snapshot(world, step)`` reads what it needs (kept in
    :attr:`snapshots`; a step entered while no time passed is read at that tick too), and every tick
    ``each_tick(world, step)`` runs. ``results`` holds what the plan reports back (command
    feedback); the test never acts on the plan's behalf.
    """

    def __init__(self, snapshot=None, each_tick=None) -> None:
        from collections import deque

        self.current, self.snapshots, self.results = "start", {}, {}
        self._snapshot, self._each_tick, self._entered = snapshot, each_tick, deque(["start"])

    def enter(self, name: str) -> None:
        """The plan starts step ``name`` (a fresh name per step)."""
        self.current = name
        self._entered.append(name)

    def __call__(self, world) -> None:
        while self._entered:
            name = self._entered.popleft()
            if self._snapshot is not None:
                self.snapshots[name] = self._snapshot(world, name)
        if self._each_tick is not None:
            self._each_tick(world, self.current)


def move_body(world, name: str, position, yaw: float = 0.0) -> None:
    """Place the kinematic scene body ``name`` (furniture, a course's box) at a world position [m] and
    yaw [rad] (test side only)."""
    import math

    from phygs.scene.assets import SceneBodies

    path = world.scene_def.bodies[name]
    SceneBodies(world.scene).write_pose(path, [*map(float, position), 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def body_box(world, name: str):
    """A scene body's collider box now: centre [m] and size [m] (world-aligned)."""
    return world.scene_def.object_box(name)


def clearance(world, boxes) -> float:
    """Signed distance [m] from the robot's unpadded transit footprint, at its base pose now, to the
    nearest of ``boxes`` ((centre, size) [m]); negative: they overlap (the robot touches it)."""
    import math

    from phygs.execution.navigation.footprint import transit_footprint
    from phygs.execution.navigation.known_map import footprint_corners, rectangle_box_distance

    footprint = getattr(world, "_robot_test_footprint", None)
    if footprint is None:
        profile = yaml.safe_load(world.robot.navigation.footprint_profile.read_text())
        footprint = transit_footprint(profile)
        world._robot_test_footprint = footprint
    pose = world.truth().base_pose()
    corners = footprint_corners(footprint, (float(pose[0, 3]), float(pose[1, 3])), math.atan2(pose[1, 0], pose[0, 0]))
    distances = [
        rectangle_box_distance(corners, np.asarray(center[:2], dtype=float), np.asarray(size[:2], dtype=float) / 2)
        for center, size in boxes
    ]
    return float(min(distances)) if distances else math.inf


def agent_checks(check: Check, report: dict, levels: tuple[str, ...] = ("skills",)) -> None:
    """The plan returned and used only the session levels ``levels`` (``skills``, ``motion``,
    ``joints``); arm contacts are recorded, not judged."""
    outcome = report["agent_outcome"]
    ok = outcome["outcome"] == "agent_returned"
    check.measure("agent_returned", outcome["outcome"], ok, outcome["detail"][-800:])
    used = report["usage"]["levels_used"]
    check.measure("levels_used", used, set(used) <= set(levels), f"the plan used levels beyond {list(levels)}")
    check.data["arm_contacts"] = report["contacts"]
    check.data["grasp_contacts"] = report["grasp_contacts"]


# Arm diagnostics (test side): what the arm rungs measure besides the plan's outcome.
ROBOT_PATH = "/World/envs/env_0/Robot"


def wrist_pose(world) -> np.ndarray:
    """The physical wrist link's world pose (4x4) [m]."""
    from scipy.spatial.transform import Rotation

    robot = world.scene["robot"]
    wrist = robot.body_names.index(world.robot.arm.wrist_link)
    pose = robot.data.body_link_pose_w.torch[0, wrist].cpu().numpy()
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = Rotation.from_quat(pose[3:]).as_matrix(), pose[:3]
    return matrix


def pose_error(achieved: np.ndarray, goal) -> tuple[float, float]:
    """Position [m] and rotation [rad] error of a 4x4 pose against a goal [x, y, z, qx, qy, qz, qw]."""
    from scipy.spatial.transform import Rotation

    goal = np.asarray(goal, dtype=float)
    rotation = (Rotation.from_matrix(achieved[:3, :3]).inv() * Rotation.from_quat(goal[3:])).magnitude()
    return float(np.linalg.norm(achieved[:3, 3] - goal[:3])), float(rotation)


def geometry_audit(world, cell_size: float) -> dict:
    """USD collider coverage by the planner's sphere model of the robot (``qualified`` or not)."""
    from phygs.execution.manipulation.planners.model import audit_collision_geometry, collider_spheres

    planning = world.robot.planning
    spheres = collider_spheres(world.sim.stage, ROBOT_PATH, planning, cell_size=cell_size)
    return audit_collision_geometry(world.sim.stage, ROBOT_PATH, {"spheres": spheres}, planning)


def link_alignment(world) -> dict:
    """The planner model's links at the measured joints against the physical links: the largest
    position [m] and rotation [rad] error, and the link with the largest position error."""
    from scipy.spatial.transform import Rotation

    from phygs.execution.manipulation.planners.model import link_transforms

    robot = world.scene["robot"]
    packet = world.control.observe()
    joints = dict(zip(packet.joints.names, packet.joints.positions.cpu().tolist(), strict=True))
    body = packet.localization.pose.matrix.cpu().numpy()
    worst = dict(position_m=0.0, rotation_rad=0.0, link=None)
    for name, frame in link_transforms(world.planner.model["urdf"], joints).items():
        if name not in robot.body_names:
            continue
        actual = robot.data.body_link_pose_w.torch[0, robot.body_names.index(name)].cpu().numpy()
        predicted = body @ frame
        position = float(np.linalg.norm(predicted[:3, 3] - actual[:3]))
        rotation = float((Rotation.from_matrix(predicted[:3, :3]).inv() * Rotation.from_quat(actual[3:])).magnitude())
        if position > worst["position_m"]:
            worst.update(position_m=round(position, 5), link=name)
        worst["rotation_rad"] = round(max(worst["rotation_rad"], rotation), 5)
    return worst


class CameraTiming:
    """Every capture of the named cameras, from the per-tick call: each frame valid, at the robot's
    resolution, and no capture period skipped (:meth:`result`)."""

    def __init__(self, world, names: tuple[str, ...]) -> None:
        cameras = world.robot.cameras
        self.names, self.period = names, world.settings["camera_period"]
        self.shapes = {name: (cameras[name].height, cameras[name].width) for name in names}
        self.stamps = {name: [] for name in names}
        self.invalid: list[str] = []

    def __call__(self, world) -> None:
        packet = world.control.observe()
        for name in self.names:
            camera = packet.cameras[name]
            stamp = camera.stamp.time_seconds
            if self.stamps[name] and stamp == self.stamps[name][-1]:
                continue
            self.stamps[name].append(stamp)
            if not (camera.valid and tuple(camera.depth.shape) == self.shapes[name]) and len(self.invalid) < 20:
                self.invalid.append(f"{name} at {stamp:.3f} s")

    def result(self) -> dict:
        gaps = [b - a for stamps in self.stamps.values() for a, b in zip(stamps, stamps[1:])]
        return dict(
            frames={name: len(stamps) for name, stamps in self.stamps.items()},
            max_gap_s=round(max(gaps), 6) if gaps else None,
            invalid=self.invalid,
        )


def first_tool_pose(folder: Path) -> np.ndarray:
    """The first tool pose the agent read in the episode (``manipulation.tool_pose`` in
    ``folder/audit.jsonl``), as a world 4x4 [m]."""
    import json

    from phygs_robot_tests.routines.geometry import matrix

    with (folder / "audit.jsonl").open() as audit:
        for line in audit:
            entry = json.loads(line)
            if entry["call"] == "manipulation.tool_pose":
                return matrix(entry["result"])
    raise ValueError(f"The agent never read a tool pose ({folder / 'audit.jsonl'}).")


def unintended_contacts(report: dict, allowed: tuple[str, ...] = ()) -> list[dict]:
    """The report's arm contacts outside a grasp with any body but ``allowed`` (e.g. the target)."""
    return [record for record in report["contacts"] if record["body"] not in allowed]


def save_view(rgb: np.ndarray, path: Path, label: str) -> None:
    """A camera frame with a crosshair at the image centre (the optical axis) and a label."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    x, y = image.width // 2, image.height // 2
    draw.line((x - 20, y, x + 20, y), fill="red", width=2)
    draw.line((x, y - 20, x, y + 20), fill="red", width=2)
    draw.text((10, 10), label, fill="red")
    image.save(path)


def blocked_start(world, goal: dict[str, float]) -> dict:
    """Ask the arm planner for a plan from the measured arm, inside a 4 m box around the robot, to
    the arm with the ``goal`` joints [rad] changed; its result (``success`` must be False)."""
    from phygs.execution.manipulation.planners.interface import PlanRequest

    packet = world.control.observe()
    joints = dict(zip(packet.joints.names, packet.joints.positions.cpu().tolist(), strict=True))
    names = world.robot.arm.joints  # Chain order.
    target = {**joints, **goal}
    request = PlanRequest(
        start=[joints[name] for name in names],
        goal=[target[name] for name in names],
        boxes=[dict(size=[4.0, 4.0, 4.0], pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
    )
    return world.planner.worker.submit(request).result(timeout=120.0)


def jaw_bottom(world) -> float:
    """The lowest point [m] of the jaws' collision spheres in the planner's model, at the measured joints."""
    from phygs.execution.manipulation.planners.model import transformed_spheres

    model = world.planner.model
    packet = world.control.observe()
    joints = dict(zip(packet.joints.names, packet.joints.positions.cpu().tolist(), strict=True))
    jaws = {name: model["spheres"][name] for name in world.robot.gripper.jaw_links}
    spheres = transformed_spheres({**model, "spheres": jaws}, joints, packet.localization.pose.matrix.cpu().numpy())
    return float(np.min(spheres[:, 2] - spheres[:, 3]))


# Navigation diagnostics (test side).
def physical_sample(world):
    """The robot's ground-truth state for a navigation evaluator (``PhysicalSample``)."""
    from physical import physical_sample as sample

    return sample(world.control, world.scene["robot"])


def navigation_evaluator(spec, goal, tolerance, budget_s: float, now_s: float, fallen_height: float | None):
    """A ``PointNavigationEvaluator`` for ``goal`` (x, y [m], heading [rad]) with ``tolerance``
    (position [m], heading [rad]), ending ``budget_s`` [s] after ``now_s``: it judges arrival as the
    robot within tolerance and stopped, from ground truth, and a fall below the robot's
    ``fallen_height`` [m] (None: it cannot fall)."""
    from phygs_robot_tests.tasks.point_navigation import PointNavigationEvaluator, PointNavigationTask

    from phygs.skills.navigation.api import NavigationGoal

    task = PointNavigationTask(
        scene_id=spec.scene.name,
        initial_state_id=spec.id,
        spawn=tuple(spec.robot.start),
        goal=NavigationGoal(*goal, *tolerance),
        time_budget=now_s + budget_s,
        minimum_height=fallen_height,
    )
    return PointNavigationEvaluator(task)


def footprint_matches(world) -> bool:
    """The robot's collision geometry is the one the transit footprint profile was measured on."""
    from phygs.sim.footprint import CollisionFootprint

    profile = yaml.safe_load(world.robot.navigation.footprint_profile.read_text())
    geometry = CollisionFootprint(world.sim.stage, world.scene["robot"], ROBOT_PATH, base=world.base)
    return geometry.geometry_sha256 == profile["geometry_sha256"]


def box_cells_marked(nav2, center, size) -> list[float | None]:
    """The fraction of a box's footprint cells each Nav2 costmap (global, local) holds at 253 or 254
    (``nav2``: the navigation backend, a trusted diagnostic). The local costmap is a window on the
    robot: only the box's cells inside it count (None when none is)."""
    from nav2_msgs.srv import GetCostmap

    transport = nav2._transport
    fractions = []
    for client in transport._costmaps:
        costmap = transport._wait(client.call_async(GetCostmap.Request())).map
        meta = costmap.metadata
        cells = np.asarray(costmap.data, dtype=np.uint8).reshape(meta.size_y, meta.size_x)
        origin = (meta.origin.position.x, meta.origin.position.y)
        lower = np.floor((np.subtract(center[:2], np.divide(size[:2], 2)) - origin) / meta.resolution)
        upper = np.ceil((np.add(center[:2], np.divide(size[:2], 2)) - origin) / meta.resolution)
        (x0, y0), (x1, y1) = np.maximum(lower.astype(int), 0), upper.astype(int)
        inside = cells[y0:y1, x0:x1]
        fractions.append(round(float(np.mean((inside >= 253) & (inside < 255))), 3) if inside.size else None)
    return fractions
