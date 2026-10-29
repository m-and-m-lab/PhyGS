# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The arm's collision-avoiding planner (cuRobo or MoveIt worker process).

Skills ask for plans through :meth:`ArmPlanner.request` from inside the simulation tick and get
a Future; the host calls :meth:`ArmPlanner.poll` every tick to carry requests out. The planner
plans the arm joints; the legs are locked joints of its model. Each request carries the legs as
measured, and the worker moves them in its model in place, so one worker serves the whole
episode however the base walks.

Obstacles come from the robot's own depth cameras (``cameras``), from known boxes (``boxes``, a
privileged source given only when the task grants it), or both. Each plan builds a new depth map
in the body frame from the cameras' first images after the request, taken with the arm still; the
planning library removes the robot's and the held object's pixels itself (see
:class:`~phygs.execution.manipulation.planners.interface.PlanRequest`).
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from phygs.execution.manipulation.collision_map import boxes_in_body
from phygs.execution.manipulation.planners.interface import DepthImage, HeldShape, PlanRequest
from phygs.execution.manipulation.planners.model import gripper_positions, link_transforms, prepare_model
from phygs.execution.manipulation.planning import JointTrajectory, PlannerProcess
from phygs.recording.panels import PlannerView


class ArmPlanner:
    """One planner worker for the episode, started after reset, for ``robot`` (its description: the
    arm, the gripper, the planning model's settings and the depth map's work volume in the body frame,
    outside which depth is not used).

    ``settings``: ``backend`` (``curobo`` or ``moveit``), ``curobo_dependencies`` (cuRobo's Python
    path), ``planning_time_seconds``, ``obstacle_padding`` (clearance [m]), ``held_sphere_limit``
    (cuRobo's reserved held-object spheres), ``planner_resolution`` [m], ``pixel_stride`` (MoveIt
    fuses every stride-th pixel per axis) and ``self_filter_padding`` (depth within this of the
    robot or its held object is dropped [m]).

    ``observe`` returns the observation packet the planner reads: joints, localization and the
    ``cameras``. ``geometry`` and ``usd_boxes`` are the robot's collision spheres and USD-only
    parts (see :func:`~phygs.execution.manipulation.planners.model.prepare_model`).
    ``boxes(exclude)`` returns known obstacle boxes {name: (center, size)} [m] in the world frame,
    without the named bodies; ``target`` is the body left out while the robot holds an unnamed object.

    ``stow`` (``stow_posture``, ``prestow_posture`` [m or rad, arm joints], ``move_seconds`` [s],
    ``tolerance`` [rad]): a plan asked with the arm within ``tolerance`` of the stow starts at the
    pre-stow posture (sphere models call the folded stow a self-collision, so planners reject it as a
    start) and its trajectory first moves straight there over ``move_seconds``, as ``unstow`` does.
    """

    def __init__(
        self,
        robot,
        *,
        project: Path,
        output: Path,
        settings: dict,
        geometry: dict,
        usd_boxes: dict,
        observe: Callable[[], Any],
        episode_id: Callable[[], int],
        time_seconds: Callable[[], float],
        cameras: tuple[str, ...] = (),
        boxes: Callable[[frozenset[str]], dict[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        target: str | None = None,
        stow: dict | None = None,
    ) -> None:
        if not cameras and boxes is None:
            raise ValueError("The arm planner needs depth cameras, known boxes, or both.")
        self._project, self._output, self._settings = project, output, settings
        self._geometry, self._usd_boxes, self._observe = geometry, usd_boxes, observe
        self._cameras, self._boxes, self._target = tuple(cameras), boxes, target
        self._robot, self._episode_id, self._time = robot, episode_id, time_seconds
        self._gripper = robot.gripper.primary
        shapes = {(robot.cameras[name].height, robot.cameras[name].width) for name in robot.cameras_for("planning")}
        if len(shapes) > 1:
            raise ValueError(f"{robot.name}: the planning cameras' images differ in size {sorted(shapes)}.")
        # The depth images' [height, width] [px]; a robot without planning cameras plans on known boxes.
        self._image_shape = list(shapes.pop()) if shapes else [1, 1]
        self._lower = np.array(robot.planning.workspace_lower)
        self._upper = np.array(robot.planning.workspace_upper)
        self._stow = stow
        self.model: dict = {}  # Shared with the skills (IK reads its URDF); filled by start.
        self.worker: PlannerProcess | None = None
        self._pending: list[tuple[dict, Future]] = []
        self._images = 0
        self._plans = 0
        self._domain: tuple[int, Any] | None = None  # MoveIt's private ROS domain and its lock.
        self.last_plan: dict | None = None  # kind, seconds [wall], result of the latest finished plan.
        self.last_view: PlannerView | None = None  # The latest finished plan for display (recording.panels).
        backend = settings["backend"]
        self._env = {
            **os.environ,
            "PYTHONPATH": str(project) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        if backend == "curobo":
            self._env["PYTHONPATH"] = settings["curobo_dependencies"] + os.pathsep + self._env["PYTHONPATH"]
        elif backend != "moveit":
            raise ValueError(f"Unknown planner backend {backend!r}; expected curobo or moveit.")
        self._command = [
            "/usr/bin/python3" if backend == "moveit" else sys.executable,
            "-m",
            "phygs.execution.manipulation.planners.worker",
            "--backend",
            backend,
        ]

    @property
    def startup_seconds(self) -> float | None:
        """Worker startup [s wall]; None before :meth:`start`."""
        return None if self.worker is None else self.worker.startup_seconds

    def start(self) -> None:
        """Build the planning model with the legs as they stand and start the worker (synchronous:
        seconds of wall time)."""
        self.model.clear()
        self.model.update(
            prepare_model(
                self._robot,
                self._output,
                geometry=self._geometry,
                locked_joints=self._joints(self._observe()),
                usd_boxes=self._usd_boxes,
            )
        )
        settings = self._settings
        self.model["planning_time_seconds"] = settings["planning_time_seconds"]
        self.model["clearance"] = settings["obstacle_padding"]
        lower, upper = self._lower, self._upper
        voxel = settings["planner_resolution"]
        self.model["depth_map"] = dict(  # The planners' depth-map volume, in the body frame.
            voxel_size=voxel,
            center=((lower + upper) / 2).tolist(),
            extent=(upper - lower + 2 * voxel).tolist(),
            image_shape=self._image_shape,
            pixel_stride=settings["pixel_stride"],
            self_filter_padding=settings["self_filter_padding"],
        )
        for folder in ("depth", "world"):
            (self._output / "planner" / folder).mkdir(parents=True, exist_ok=True)
        if self._settings["backend"] == "moveit":
            # MoveIt's self filter reads the robot's link frames from /tf: the worker gets a ROS domain
            # of its own, so its frames never reach Nav2 or another planner, nor theirs reach it.
            # (Discovery off also stops the worker's own nodes from finding each other.)
            self._domain = private_ros_domain()
            self._env["ROS_DOMAIN_ID"] = str(self._domain[0])
        self.worker = PlannerProcess(self._command, self.model, self._output / "planner", self._env)

    def warm_up(self, grippers: tuple[float, ...] = (), timeout: float = 120.0) -> float:
        """Plan once with the measured gripper angle and once per angle in ``grippers`` [rad] before
        the episode starts (synchronous); returns the wall time [s]. cuRobo builds and warms one
        motion planner per gripper angle on the first plan with it (about 8 s each, 2026-10-26: a
        house episode's first three plans); warmed here, the agent's first plans take their usual
        fraction of a second."""
        started = time.perf_counter()
        packet = self._observe()
        joints = self._joints(packet)
        legs = self._legs(joints)
        start = list(self._stow["prestow_posture"]) if self._stow else [joints[name] for name in self._robot.arm.joints]
        goal = [*start[:-1], start[-1] + 0.1]  # A small wrist turn: free of any obstacle.
        for gripper in (joints[self._gripper], *grippers):
            request = PlanRequest(start=start, boxes=[], goal=goal, gripper=float(gripper), locked_joints=legs)
            self.worker.submit(request).result(timeout=timeout)
        return time.perf_counter() - started

    def status(self) -> dict:
        """The planner for telemetry: running, busy, restarts (always 0) and the latest plan."""
        return dict(
            running=self.worker is not None,
            busy=bool(self._pending) or bool(self.worker is not None and self.worker.busy),
            restarts=0,
            last_plan=self.last_plan,
        )

    # Called by the skills inside the tick.
    def request(
        self, *, goal: list[float] | None = None, world_pose: tuple[float, ...] | None = None, held=None
    ) -> Future:
        """A plan to arm joints [rad] or to a world wrist pose [m, xyzw]; resolved by :meth:`poll`.

        ``held`` (a ``HeldObject``) is the object the gripper carries, as the caller says; it is
        part of the robot for the plan, and the known box it names (else ``target``) is left out.
        A failed plan raises ``RuntimeError("Joint planner failed: <reason>")`` (or ``Pose``) from
        the Future."""
        if (goal is None) == (world_pose is None):
            raise ValueError("Plan to a joint goal or to a world pose, not both.")
        future: Future = Future()
        self._pending.append((dict(goal=goal, world_pose=world_pose, held=held, asked=self._time()), future))
        return future

    # Called by the host every tick.
    def poll(self) -> None:
        """Submit the next request once the worker is free and every camera has an image taken after it."""
        if not self._pending or self.worker is None or self.worker.busy:
            return
        request, future = self._pending[0]
        packet = self._observe()
        if any(packet.cameras[name].stamp.time_seconds < request["asked"] for name in self._cameras):
            return
        self._pending.pop(0)
        if not future.set_running_or_notify_cancel():
            return
        kind = "Joint" if request["goal"] is not None else "Pose"
        started = time.perf_counter()
        try:
            plan_request, lead = self._plan_request(packet, **request)
            pending = self.worker.submit(plan_request)
        except BaseException as error:
            future.set_exception(error)
            return
        episode = self._episode_id()

        def finish(done: Future) -> None:
            try:
                result = done.result()
                self._record(plan_request, result)
                self.last_plan = dict(
                    kind=kind.lower(),
                    seconds=time.perf_counter() - started,
                    result="planned" if result["success"] else result.get("reason"),
                )
                if not result["success"]:
                    raise RuntimeError(f"{kind} planner failed: {result.get('reason')}")
                future.set_result(self._trajectory(result, episode, lead))
            except BaseException as error:
                future.set_exception(error)
            if done.exception() is None:  # The display never decides the plan; its failure is shown.
                try:
                    self.last_view = self._view(plan_request, done.result(), request["asked"])
                except Exception as error:  # noqa: BLE001 - shown in the video's planner panel.
                    self.last_view = PlannerView(label=f"display failed: {type(error).__name__}: {error}")

        pending.add_done_callback(finish)

    def _record(self, request: PlanRequest, result: dict) -> None:
        """One line per plan in ``planner/requests.jsonl``: the request (replayable offline) and the outcome."""
        outcome = {key: result[key] for key in ("success", "reason", "plan_seconds", "map_seconds") if key in result}
        line = dict(t=round(self._time(), 3), request=request.to_json(), result=outcome)
        with (self._output / "planner" / "requests.jsonl").open("a") as file:
            file.write(json.dumps(line) + "\n")

    def _plan_request(self, packet, *, goal, world_pose, held, asked) -> tuple[PlanRequest, tuple | None]:
        """The worker's request and, from the stow, the straight lead to the pre-stow posture (start
        and end arm joints [rad], duration [s]) its trajectory begins with."""
        if packet.localization is None or not packet.localization.valid:
            raise PermissionError("The arm planner needs permitted localization.")
        world_from_body = packet.localization.pose.matrix.cpu().numpy()
        joints = self._joints(packet)
        legs = self._legs(joints)
        boxes = []
        if self._boxes is not None:
            name = getattr(held, "name", None) or self._target
            exclude = frozenset({name}) if held is not None and name else frozenset()  # Held: part of the robot.
            boxes = [
                dict(size=np.asarray(size).tolist(), pose=[*np.asarray(center).tolist(), 0.0, 0.0, 0.0, 1.0])
                for center, size in self._boxes(exclude).values()
            ]
        goal_pose = None
        if world_pose is not None:
            local = np.linalg.inv(world_from_body) @ _matrix(world_pose)
            goal_pose = [*local[:3, 3].tolist(), *Rotation.from_matrix(local[:3, :3]).as_quat().tolist()]
        world = None
        if self._settings["backend"] == "curobo" and self._cameras:
            world = str(self._output / "planner" / "world" / f"{self._plans:06d}.npy")
        self._plans += 1
        start, lead = [joints[name] for name in self._robot.arm.joints], None
        stow = self._stow
        if (
            stow is not None
            and max(abs(a - b) for a, b in zip(start, stow["stow_posture"], strict=True)) <= stow["tolerance"]
        ):
            start, lead = list(stow["prestow_posture"]), (start, list(stow["prestow_posture"]), stow["move_seconds"])
        return PlanRequest(
            start=start,
            boxes=boxes_in_body(boxes, world_from_body),
            goal=None if goal is None else list(goal),
            goal_pose=goal_pose,
            gripper=joints[self._gripper],  # Measured, e.g. on a held object.
            held=None if held is None else self._held_shape(held),
            locked_joints=legs,
            images=[self._image(packet.cameras[name], name) for name in self._cameras] or None,
            world_path=world,
        ), lead

    def _view(self, request: PlanRequest, result: dict, asked: float) -> PlannerView:
        """A finished plan for display, in the body frame: the planner's obstacle voxels (when it saved
        them), the known boxes, the arm at the start, along the plan and at the end, its collision
        spheres at the end, the jaw's path and the floor (the lowest foot sphere's bottom)."""
        legs = {**request.locked_joints, **gripper_positions(self.model, request.gripper)}
        poses = [{**legs, **dict(zip(self._robot.arm.joints, request.start, strict=True))}]
        if result["success"]:
            names, samples = result["joint_names"], result["positions"]
            step = max(1, len(samples) // 40)
            poses += [{**legs, **dict(zip(names, sample, strict=True))} for sample in [*samples[::step], samples[-1]]]
        frames = [link_transforms(self.model["urdf"], pose) for pose in poses]
        arm_links = set(self._robot.arm_links)
        arms = [np.array([pose[:3, 3] for link, pose in frame.items() if link in arm_links]) for frame in frames]
        body = self._spheres(frames[0], arm=False)
        world = request.world_path
        voxels = np.load(world) if world is not None and Path(world).exists() else np.zeros((0, 3))
        seconds = result.get("plan_seconds", 0.0)
        outcome = "planned" if result["success"] else result.get("reason", "failed")
        moved = len(arms) > 1
        return PlannerView(
            label=f"t {asked:.1f} s: {outcome} in {seconds:.2f} s; {len(voxels)} voxels",
            voxels=voxels,
            voxel_size=self._settings["planner_resolution"],
            boxes=tuple((tuple(box["pose"][:3]), tuple(box["size"])) for box in request.boxes),
            arm_start=arms[0],
            arm_path=tuple(arms[1 : -1 : max(1, (len(arms) - 2) // 8)]),
            arm_end=arms[-1] if moved else np.zeros((0, 3)),
            end_spheres=self._spheres(frames[-1], arm=True) if moved else np.zeros((0, 4)),
            tool_path=np.array([frame[self._robot.planning.drawn_link][:3, 3] for frame in frames[1:]]).reshape(-1, 3),
            floor=float((body[:, 2] - body[:, 3]).min()) if len(body) else None,
        )

    def _spheres(self, frames: dict[str, np.ndarray], *, arm: bool) -> np.ndarray:
        """The arm's (or the rest of the robot's) collision spheres [x, y, z, r] [m] at link poses ``frames``."""
        arm_links = set(self._robot.arm_links)
        rows = [
            np.column_stack(
                (
                    np.array([s["center"] for s in found]) @ frames[link][:3, :3].T + frames[link][:3, 3],
                    [s["radius"] for s in found],
                )  # fmt: skip
            )
            for link, found in self.model["spheres"].items()
            if (link in arm_links) == arm
        ]
        return np.concatenate(rows) if rows else np.zeros((0, 4))

    def _image(self, camera, name: str) -> DepthImage:
        """The camera's depth for the planner, saved under ``planner/depth``: 0 where there is no
        data or the point is outside the work volume."""
        depth = np.where(camera.depth_valid.cpu().numpy(), camera.depth.cpu().numpy(), 0.0).astype(np.float32)
        intrinsics = camera.intrinsics.cpu().numpy()
        body_from_camera = camera.extrinsics.matrix.cpu().numpy()
        rows, cols = np.indices(depth.shape).reshape(2, -1)
        rays = np.linalg.solve(intrinsics, np.stack((cols, rows, np.ones_like(rows))))
        points = body_from_camera[:3, :3] @ (rays * depth.reshape(-1)) + body_from_camera[:3, 3:]
        inside = np.all((self._lower <= points.T) & (self._upper >= points.T), axis=1).reshape(depth.shape)
        path = self._output / "planner" / "depth" / f"{self._images:06d}_{name}.npy"
        self._images += 1
        np.save(path, np.where(inside, depth, 0.0).astype(np.float32))
        pose = [*body_from_camera[:3, 3].tolist(), *Rotation.from_matrix(body_from_camera[:3, :3]).as_quat().tolist()]
        return DepthImage(path=str(path), intrinsics=intrinsics.tolist(), pose=pose)

    def _held_shape(self, held) -> HeldShape:
        """The held box (MoveIt) and covering spheres (cuRobo), cells grown until cuRobo's
        reserved held-object spheres hold them; collision model only, never attached in physics."""
        cell = self._settings["obstacle_padding"]
        while np.prod(np.maximum(1, np.ceil(np.asarray(held.size) / cell))) > self._settings["held_sphere_limit"]:
            cell *= 1.25
        return HeldShape(box=held.box(), spheres=held.spheres(np.eye(4), cell).tolist())

    @staticmethod
    def _joints(packet) -> dict[str, float]:
        return dict(zip(packet.joints.names, packet.joints.positions.cpu().tolist(), strict=True))

    def _legs(self, joints: dict[str, float]) -> dict[str, float]:
        """The measured positions of the model's locked joints other than the gripper's (the legs)."""
        gripper = self.model["gripper_ratios"]
        return {name: joints[name] for name in self.model["locked_joints"] if name not in gripper}

    def _trajectory(self, samples: dict, episode: int, lead: tuple | None = None) -> JointTrajectory:
        """Planner samples as a trajectory; one sample (already there) holds it for 32 ms. ``lead``
        (start and end arm joints [m or rad], duration [s]) is a straight move the trajectory begins with."""
        names, times, positions = tuple(samples["joint_names"]), samples["times"], samples["positions"]
        if len(times) == 1:
            times, positions = [0.0, 0.032], positions * 2
        if lead is not None:
            start, end, seconds = lead
            order = [self._robot.arm.joints.index(name) for name in names]
            steps = max(2, round(seconds / 0.032))
            fractions = np.linspace(0.0, 1.0, steps + 1)
            straight = np.asarray(start)[order] + fractions[:, None] * (
                np.asarray(end)[order] - np.asarray(start)[order]
            )
            times = [*(fractions * seconds).tolist(), *(seconds + np.asarray(times[1:])).tolist()]
            positions = [*straight.tolist(), *list(positions[1:])]
        return JointTrajectory(names, times, positions, episode)

    def close(self) -> None:
        for _, future in self._pending:
            future.cancel()
        self._pending.clear()
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self._domain is not None:
            self._domain[1].close()  # Frees the domain.
            self._domain = None


def private_ros_domain() -> tuple[int, Any]:
    """A ROS domain in 215-232 (unused by convention) no other planner worker on this machine holds,
    and its lock file, held open until closed."""
    for domain in range(215, 233):
        lock = open(f"/tmp/phygs_ros_domain_{domain}.lock", "w")  # noqa: SIM115 - held.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return domain, lock
        except BlockingIOError:
            lock.close()
    raise RuntimeError("Every private ROS domain (215-232) is held by another MoveIt planner worker.")


def _matrix(pose) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix
