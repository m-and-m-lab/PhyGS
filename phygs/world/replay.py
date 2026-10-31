# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a finished run again from its state log (``phygs replay``), so the run itself renders nothing.

Needs a running Isaac Sim app (``AppLauncher`` before import). :class:`Replay` builds the run's
scene and robot again, without the robot's control, Nav2 or the arm planner, then for each sample
of the state log (:mod:`phygs.recording.state_log`) writes the robot's root pose and joints and
every scene body's pose, updates the kinematics without stepping physics, and renders: a camera
outside that follows the robot, and the robot's own cameras. With the logged costmap, route,
commanded velocity and arm plans, a frame has the recording's panel sources
(:class:`~phygs.recording.panels.PanelSources`), so the replay can draw the recording's panels
beside the outside view. :func:`render_plan` renders a scene's floor plan for
:mod:`phygs.scene.plan`'s cache.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.scene import InteractiveScene

from phygs.observations.front_view import CameraFrame, FrontStitcher, depth_colors
from phygs.recording.panels import (
    DEPTH_RANGE,
    NavigationView,
    PanelFeed,
    PanelSources,
    navigation_image,
    planner_image,
)
from phygs.recording.state_log import States
from phygs.recording.video import FOCAL_LENGTH, Chase
from phygs.robots.description import RobotDescription, load_robot
from phygs.robots.scene import make_scene_cfg, robot_cameras, simulation_cfg
from phygs.scene import plan as plans
from phygs.scene.assets import SceneBodies
from phygs.scene.reader import UsdScene
from phygs.scene.registry import scene_file
from phygs.sim.base import BaseLink
from phygs.sim.cameras import RenderCameras, ViewCamera
from phygs.world import build

ROBOT_PATH = "/World/envs/env_0/Robot"
PLAN_CUT = 2.4  # [m] above a floor: a plan leaves out everything higher (the ceiling, the floor above).
PLAN_EYE = 40.0  # [m] above a floor: a plan is rendered from this high, straight down.
PLAN_WIDTH = 1800  # [px]
PLAN_MARGIN = 1.0  # [m] around the scene's map area.
VIEWS = ("front", "front_depth", "hand", "hand_depth", "navigation", "arm_planner")  # One clip each.


class Replay:
    """The run in folder ``run`` (its ``state.npz``), ready to render sample by sample.

    ``robot`` replaces the run's robot (a description, a name or a ``robot.yaml``); it must have the
    same joints. ``views`` are the robot's views to render (:data:`VIEWS`; the cameras are rendered
    for the ``front`` and ``hand`` ones); ``outside`` the size (width, height) [px] of the outside
    camera (default: the settings' ``record`` size, as the recording's), None for none; ``fixed``
    keeps it at the run's record view instead of behind the robot (recording.video's ``Chase``).
    """

    def __init__(
        self,
        run: Path,
        *,
        robot: RobotDescription | str | Path | None = None,
        views: tuple[str, ...] = VIEWS,
        outside: tuple[int, int] | None | str = "record",
        fixed: bool = False,
        device: str = "cpu",
    ) -> None:
        self.states = states = States.load(run)
        meta = states.meta
        if not isinstance(robot, RobotDescription):
            robot = load_robot(meta["robot"] if robot is None else robot)
        self.robot = robot
        self.period, self.dt = float(meta["period"]), float(meta["physics_dt"])
        scene = meta["scene"]
        usd = Path(scene["usd"])
        if not usd.is_file():  # Another machine: the same release from this one's scene cache.
            usd = scene_file(scene["id"], scene["version"])
        settings = build.load_settings(build.SETTINGS)
        x, y, heading = meta["start"]
        self.scene_def = UsdScene(
            scene["name"],
            usd,
            ground=scene["ground"],
            objects=meta["objects"],
            navigation_map=settings["map"],
            robot_start=meta["start"],
        )
        self.scene_def.open_stage()
        self.sim = sim_utils.SimulationContext(simulation_cfg(dt=self.dt, render_interval=1, device=device))
        front, look = robot.cameras_for("front"), robot.look_camera
        self._front = front if any(view.startswith("front") for view in views) else ()
        self._look = look if any(view.startswith("hand") for view in views) else None
        names = tuple(name for name in PanelFeed.cameras(robot) if name in (*self._front, self._look))
        cfg = make_scene_cfg(
            robot,
            physics_dt=self.dt,
            light=meta["light"],
            position=(x, y, robot.spawn_height),
            heading=heading,
            cameras=robot_cameras(robot, names),
        )(num_envs=1, env_spacing=3.0, replicate_physics=False)
        self.scene_def.define(self.sim.stage, cfg, self.dt, contact_links=robot.arm_links)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.scene_def.bind(self.scene)
        self._robot = self.scene["robot"]
        self.base = BaseLink(self._robot, robot.base_link)
        self._bodies = SceneBodies(self.scene)
        missing = set(states.joint_names) - set(self._robot.joint_names)
        if missing:
            raise ValueError(f"The robot {robot.name} lacks the run's joints {sorted(missing)}.")
        self._joint_ids = [self._robot.joint_names.index(name) for name in states.joint_names]
        index = {name: i for i, name in enumerate(states.body_names)}
        missing = [name for name in self._bodies.names if name not in index]
        if missing:
            raise ValueError(f"The state log has no poses for {len(missing)} scene bodies, e.g. {missing[0]}.")
        self._order = np.asarray([index[name] for name in self._bodies.names])
        self.cameras = RenderCameras(self.scene, names) if names else None
        self._stitcher = FrontStitcher(width=640) if len(self._front) == 2 else None
        self._footprint = build.transit_outline(robot)
        self.view = None
        if outside == "record":
            outside = (settings["record"]["width"], settings["record"]["height"])
        if outside is not None:
            self.view = ViewCamera(
                "/World/ReplayCamera", width=outside[0], height=outside[1], focal_length=FOCAL_LENGTH
            )
        self._chase = Chase(self.period, eye=meta["record_view"][0] if fixed else None)

    # The scene at a sample.
    def apply(self, sample: int, bodies: np.ndarray) -> None:
        """Pose the robot and the scene's bodies [B, 7] (the log's order) as at ``sample``."""
        device = self._robot.device
        self._robot.write_root_link_pose_to_sim_index(
            root_pose=torch.as_tensor(self.states.root[sample], dtype=torch.float32, device=device).reshape(1, 7)
        )
        self._robot.write_joint_position_to_sim_index(
            position=torch.as_tensor(self.states.joints[sample], dtype=torch.float32, device=device).reshape(1, -1),
            joint_ids=self._joint_ids,
        )
        poses = torch.as_tensor(bodies[self._order], dtype=torch.float32, device=device)
        self._bodies.write_all(poses, torch.zeros((len(poses), 6), dtype=torch.float32, device=device))
        self.sim.forward()  # Kinematics and Fabric, for rendering; physics does not step.
        self.scene.update(self.dt)  # Fresh asset data (the cameras' mounts read it).

    def aim(self) -> None:
        """Aim the outside camera at the robot as posed now, as the recording does (``Chase``)."""
        self.view.place(*self._chase.aim(self.base.pose().cpu().numpy()))

    def render(self, sample: int) -> tuple[np.ndarray | None, PanelSources]:
        """Render once, posed as at ``sample`` (:meth:`apply`): the outside image (None without one)
        and the recording's panel sources: the robot's cameras, navigation (Nav2's costmap and route,
        the commanded velocity) and the arm plan, as logged then."""
        RenderCameras.render()
        outside = self.view.image() if self.view is not None else None
        pose = self.base.pose().cpu().numpy()
        front, hand = (), None
        if self.cameras is not None:
            body = _matrix(pose)
            front = tuple(self._frame(name, body) for name in self._front)
            hand = self._frame(self._look, body) if self._look else None
        return outside, PanelSources(
            front, hand, self._navigation(sample, pose), self.states.plan_at(self._time(sample))
        )

    def views(self, sources: PanelSources, outside: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """The separate views (:data:`VIEWS`, and ``outside``) of a frame's panel sources, full size."""
        views: dict[str, np.ndarray] = {} if outside is None else {"outside": outside}
        if len(sources.front) == 2 and self._stitcher is not None:
            joined = self._stitcher.stitch(*sources.front)
            depth = joined.depth()
            views["front"], views["front_depth"] = joined.rgb, depth_colors(depth, np.isfinite(depth), DEPTH_RANGE)
        elif len(sources.front) == 1:
            frame = sources.front[0]
            views["front"], views["front_depth"] = frame.rgb, depth_colors(frame.depth, frame.valid, DEPTH_RANGE)
        if sources.hand is not None:
            frame = sources.hand
            views["hand"], views["hand_depth"] = frame.rgb, depth_colors(frame.depth, frame.valid, DEPTH_RANGE)
        if sources.navigation is not None:
            views["navigation"] = navigation_image(sources.navigation, 640, 480)
        if sources.planner is not None:
            views["arm_planner"] = planner_image(sources.planner, 640, 480)
        return views

    def _time(self, sample: int) -> float:
        return float(self.states.time[sample])

    def _navigation(self, sample: int, pose: np.ndarray) -> NavigationView | None:
        """Navigation as the recording's panel showed it at ``sample``; None for a run without Nav2."""
        states, now = self.states, self._time(sample)
        if not states.routes and not len(states.costmap_times):
            return None
        costmap, route = states.costmap_at(now), states.route_at(now)
        costs, origin, cell = costmap if costmap is not None else (None, (0.0, 0.0), 0.05)
        command = states.command[sample]
        x, y, z, w = pose[3:]
        return NavigationView(
            pose=(float(pose[0]), float(pose[1]), math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))),
            costs=costs,
            origin=origin,
            resolution=cell,
            footprint=self._footprint,
            path=() if route is None else tuple(map(tuple, route.points.tolist())),
            goal=None if route is None else route.goal,
            command=None if np.isnan(command).any() else tuple(map(float, command)),
        )

    def _frame(self, name: str, world_from_body: np.ndarray) -> CameraFrame:
        frame = self.cameras.read(name)
        rgb = np.asarray(frame.rgb.cpu().numpy()[..., :3], dtype=np.uint8)
        depth = np.asarray(frame.depth.cpu().numpy(), dtype=np.float32).reshape(rgb.shape[:2])
        world_from_camera = np.asarray(frame.world_from_camera.cpu().numpy(), dtype=float)
        return CameraFrame(
            name=name,
            time_seconds=0.0,
            tick=0,
            rgb=rgb,
            depth=depth,
            valid=np.isfinite(depth) & (depth > 0),
            intrinsics=np.asarray(frame.intrinsics.cpu().numpy(), dtype=float),
            body_from_camera=np.linalg.inv(world_from_body) @ world_from_camera,
            world_from_camera=world_from_camera,
        )

    # The scene's plan.
    def plan(self, cache: Path | None = None) -> plans.Plan:
        """The scene's plan with the floor the robot started on (:func:`scene_plan`)."""
        elevation = round(float(self.states.root[0, 2]) - self.robot.spawn_height, 1)
        return scene_plan(self.scene_def, self.states.meta["scene"]["sha256"], elevation, cache=cache)

    def close(self) -> None:
        if self.cameras is not None:
            self.cameras.close()
        if self.view is not None:
            self.view.close()


class Clip:
    """An H.264 mp4 at ``fps``, written frame by frame; a missing frame (None) repeats the last one."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path, self.fps = Path(path), fps
        self._writer, self._last, self._missing = None, None, 0

    def write(self, image: np.ndarray | None) -> None:
        if image is None:
            if self._last is None:
                self._missing += 1  # Written as the first image, once there is one.
                return
            image = self._last
        image = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        image = image[: image.shape[0] - image.shape[0] % 2, : image.shape[1] - image.shape[1] % 2]  # H.264: even.
        if self._writer is None:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(
                self.path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                pixelformat="yuv420p",
                ffmpeg_params=["-movflags", "+faststart"],
            )
            for _ in range(self._missing):
                self._writer.append_data(image)
        self._writer.append_data(image)
        self._last = image

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def scene_plan(scene_def: UsdScene, sha256: str, elevation: float = 0.0, *, cache: Path | None = None) -> plans.Plan:
    """The open scene's plan with the floor at ``elevation`` [m]: the release's (as it ships), the
    cached one, or one rendered now (:func:`render_plan`) and cached under the scene's ``sha256``."""
    folder = plans.cached_folder(sha256, cache=cache)
    found = plans.find(scene_def.usd, sha256, cache=cache)
    if found is not None and (
        found.folder != folder or any(abs(floor.elevation - elevation) < 0.5 for floor in found.floors)
    ):
        return found
    name = "ground" if abs(elevation) < 0.5 else f"floor_{elevation:+.1f}".replace(".", "_")
    floor = render_plan(scene_def.map["bounds"], folder, name, elevation)
    floors = tuple(f for f in found.floors if f.name != name) if found is not None else ()
    plans.Plan(folder, (*floors, floor)).save()
    return plans.Plan.load(folder)


def render_plan(bounds, folder: Path, name: str, elevation: float = 0.0) -> plans.Floor:
    """Render one floor's plan of the open stage, straight down over the map ``bounds`` (xmin, ymin,
    xmax, ymax) [m], without the robot, into ``folder/<name>.png``; returns the floor."""
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom

    xmin, ymin, xmax, ymax = bounds
    aspect = (ymax - ymin + 2 * PLAN_MARGIN) / (xmax - xmin + 2 * PLAN_MARGIN)
    width, height = PLAN_WIDTH, 2 * round(PLAN_WIDTH * aspect / 2)
    center, ppm = plans.fit(bounds, width, height, PLAN_MARGIN)
    focal = PLAN_EYE * 20.955 * ppm / width  # [mm]: the floor plane spans width / ppm [m].
    robot = UsdGeom.Imageable(omni.usd.get_context().get_stage().GetPrimAtPath(ROBOT_PATH))
    robot.MakeInvisible()
    camera = ViewCamera(f"/World/PlanCamera_{name}", width=width, height=height, focal_length=focal)
    try:
        camera.place(
            (center[0], center[1], elevation + PLAN_EYE), (center[0], center[1], elevation), near=PLAN_EYE - PLAN_CUT
        )
        for _ in range(60):  # Textures stream in over the first renders.
            RenderCameras.render()
        image = camera.image()
        if image is None:
            raise RuntimeError("The plan camera delivered no image.")
    finally:
        camera.close()
        robot.MakeVisible()
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(folder / f"{name}.png")
    return plans.Floor(name, f"{name}.png", elevation, elevation + PLAN_CUT, width, height, center, ppm)


def _matrix(pose: np.ndarray) -> np.ndarray:
    """World 4x4 of a pose [x, y, z, qx, qy, qz, qw]."""
    x, y, z, w = pose[3:]
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = pose[:3]
    return matrix
