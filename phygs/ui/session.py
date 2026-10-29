# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot UI as an episode's agent and viewer: the page's requests become session calls, and
the episode's view becomes the page's frames, map, state and step lines.

:class:`UiSession` has two sides. As the :class:`~phygs.world.World`'s
agent (:meth:`UiSession.agent`, on the agent thread) it carries out the person's requests through
the :class:`~phygs.session.session.AgentSession`: every call is the agent API's, audited
like any agent's, within the spec's tier and grants. As its viewer (:meth:`UiSession.update`, on the
main thread once per camera period) it publishes what the person sees from the read-only
:class:`~phygs.world.viewer.EpisodeView`. It never passes simulation objects to the
server.
"""

from __future__ import annotations

import collections
import enum
import json
import math
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phygs.execution.manipulation.planning import EndEffectorTrajectory, JointTrajectory
from phygs.execution.manipulation.skills.types import HeldObject
from phygs.execution.navigation.types import NavigationGoal
from phygs.observations.front_view import camera_frame
from phygs.session import EpisodeEnded
from phygs.ui import translate
from phygs.ui.bridge import Bridge, MapImage

# Arm links drawn for a previewed plan, shoulder to wrist; the grasp point follows.
PREVIEW_SAMPLES = 40  # Points of the gripper path sent to the page.
CAMERA_WINDOW_S = 2.0  # Sim time [s] over which a camera's frame rate is measured.
AGENT_PERIOD_S = 0.05  # Wall time [s] between the agent's passes over the page's requests.
# Skill step lines the UI shows; "check" lines come from robot tests, not skills.
STEP_KINDS = ("manipulation", "navigation", "locomotion")
# Command status -> UI state (docs/design/ui/design.html, section 6). Anything else is a failure.
UI_STATES = {
    "queued": "executing",
    "active": "executing",
    "stopping": "executing",
    "succeeded": "succeeded",
    "done": "succeeded",
    "cancelled": "cancelled",
    "overridden": "cancelled",
}
CHANNELS = {"locomotion": "base", "navigation": "base", "manipulation": "arm"}
BASE_REQUESTS = ("navigate_to", "drive", "stop", "cancel")
# Episode phases on the page: the task is shown until Start; the simulation runs only while running.
READY, RUNNING, ENDED = "ready", "running", "ended"


def ui_state(status: str) -> str:
    """The UI state (executing, succeeded, cancelled or failed) of a raw command status."""
    return UI_STATES.get(status, "failed")


def plain(value: Any) -> Any:
    """Feedback as JSON-ready values: dataclasses as dicts, enums as their values, tuples as lists."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class UiSession:
    """The person at the page: an agent that carries out the page's requests, and a viewer.

    Viewer side (:class:`~phygs.world.viewer.Viewer`): ``cameras`` are the camera
    names to stream; :meth:`update` publishes frames, the step lines, navigation's costmap (shown
    every ``map_period`` [s]: free, obstacle, unknown, and where the robot's centre may not be),
    and the state with ``telemetry`` for the page's telemetry drawer: base velocity (body frame)
    and turn rate, arm joint speeds, the hand position, the running arm command's latest feedback,
    each camera's frame rate and age, the counts of commands and failures, and ``health()`` (the
    runner's timing and the planner). Every ``telemetry_period`` [s of sim time] the telemetry is
    also appended to ``telemetry_path`` (JSON lines). ``stow_tolerance`` [m or rad] reports whether
    the arm is within it of the robot's stow.

    Agent side (:meth:`agent`): requests from the page are ``episode`` (start or end), ``run`` (a
    skill from :mod:`phygs.ui.schema` with its inputs), ``preview``, ``drive`` (v_x,
    v_y, v_rot, renewed while a key is held, within ``drive_limits``), ``cancel`` (a UI command id)
    and ``estop``. Page inputs become API arguments in :mod:`phygs.ui.translate`; a call
    the API rejects becomes a failed command with the raw error. ``robot`` is the robot's
    description: its arm joints and links, its gripper, its stow, its ``hand`` tool and its
    ``grasp`` tool (where a clicked point goes, for wrist poses built from a click). The person does
    not say what the gripper holds: while
    it holds something, arm moves plan with a ``held_object_size_m`` [m] box between the jaws.
    With ``forward`` (joints [m or rad] -> body-frame link poses), ``preview`` plans Arm joints or Move
    to pose without moving (``manipulation.plan``); the page shows the plan and can run exactly that
    trajectory.

    The episode is ``ready`` until the person presses Start (the World holds the clock until
    the agent's first call), ``running``, then ``ended``: by End, E-stop (the base stops, arm
    commands are cancelled, the episode finishes), the episode's budget or a failure. ``task`` is
    the public task (goal, description) shown with ``budget_seconds`` [s of sim time]. The
    World's report is the episode's; :meth:`summary` adds the page's command counts.
    ``demo`` (session -> iterator of steps) runs a scripted route as soon as the episode starts.
    The page's requests are read every pass of the agent (``AGENT_PERIOD_S``); the commands'
    feedback, which is an audited call each, and the demo's steps every ``feedback_period`` [s wall].
    """

    def __init__(
        self,
        bridge: Bridge,
        *,
        robot,
        cameras: tuple[str, ...],
        map_period: float = 0.5,
        stow_tolerance: float = 0.2,
        drive_limits: dict | None = None,
        forward: Callable[[dict], dict] | None = None,
        held_object_size_m: tuple[float, float, float] | None = None,
        task: dict | None = None,
        budget_seconds: float | None = None,
        health: Callable[[], dict] | None = None,
        telemetry_path: Path | None = None,
        telemetry_period: float = 0.2,
        feedback_period: float = 0.25,
        demo: Callable[[Any], Iterator[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bridge, self.cameras = bridge, tuple(cameras)
        self._map_period, self._last_costs = map_period, None
        self._drive_limits = drive_limits or dict(max_speed=1.0, max_yaw_speed=1.5)
        self._arm_joints, self._arm_links, self._wrist_link = (
            robot.arm.joints,
            robot.arm_joint_links,
            robot.arm.wrist_link,
        )
        self._gripper_joint, self._hand = robot.gripper.primary, robot.tool("hand")
        self._grasp = robot.tool("grasp")  # Where Move to pose puts a clicked point, in the wrist link.
        self._held_size = None if held_object_size_m is None else tuple(map(float, held_object_size_m))
        self._stow, self._stow_tolerance = robot.arm.stow, stow_tolerance
        self._forward, self._demo_script, self._demo = forward, demo, None
        self._task, self._budget, self._health = dict(task or {}), budget_seconds, health
        self._clock, self._started = clock, clock()
        self._telemetry_period = telemetry_period
        # Each feedback read is an audited agent call: polled at 20 Hz a 3-minute run logged ~1,600.
        self._feedback_period, self._last_feedback = feedback_period, -math.inf
        self._telemetry_file = None if telemetry_path is None else telemetry_path.open("a")
        self._telemetry_written = -math.inf
        # Shared by the agent thread (commands) and the main thread (state): under the lock.
        self._lock = threading.RLock()
        self._commands: dict[str, dict] = {}  # UI command id -> its record.
        self._next_command = 1
        self._previews: dict[str, dict] = {}  # Preview id -> skill, inputs, the plan (a Future) and its trajectory.
        self._counted: set[str] = set()  # UI command ids whose end is counted.
        self._counts: dict[str, dict[str, int]] = {}  # skill -> UI state -> commands.
        self._failures: dict[str, dict[str, int]] = {}  # skill -> failure code -> commands.
        self._failure_list: list[dict] = []
        self._drive_seconds = 0.0
        self._drive: dict | None = None  # The Drive command being renewed.
        self.phase = READY
        self._start_wall: float | None = None
        self._end_wall: float | None = None
        self._outcome: tuple[str, str] | None = None  # (outcome, reason) once ended.
        self._finish_reason: str | None = None  # End or E-stop: the episode finishes on the agent's next pass.
        # Viewer side (main thread).
        self._time = 0.0  # Simulated time [s] at the last view.
        self._episode_id: int | None = None
        self._last_ticks: dict[str, int] = {}
        self._camera_times: dict[str, collections.deque[float]] = {}  # Capture times [s] per streamed camera.
        self._timing: collections.deque[tuple[float, float]] = collections.deque()
        self._last_yaw: tuple[float, float] | None = None  # (sim time [s], yaw [rad]) at the last state.
        self._last_map_wall, self._map_version = -math.inf, None
        self._map_error: str | None = None
        self._steps_seen = 0

    def bind(self, *, forward: Callable[[dict], dict] | None = None, health: Callable[[], dict] | None = None) -> None:
        """Late parts, once the episode is built: the robot's kinematic model (``forward``: joints
        [rad] -> body-frame link poses) and the runner's ``health()``."""
        if forward is not None:
            self._forward = forward
        if health is not None:
            self._health = health

    # ------------------------------------------------------------------------------------------
    # Agent side (agent thread): every robot call goes through the session.
    def agent(self, session: Any) -> None:
        """The World's agent: serve the page until the episode ends."""
        while self.agent_step(session):
            time.sleep(AGENT_PERIOD_S)

    def agent_step(self, session: Any) -> bool:
        """One pass: the page's requests, the commands' feedback, finished previews and the demo.
        Returns False once the episode has ended."""
        try:
            self._handle_requests(session)
            due = self._clock() - self._last_feedback >= self._feedback_period
            if due:
                self._last_feedback = self._clock()
            if self.phase != ENDED:
                now = self._clock()
                with self._lock:
                    records = [
                        r
                        for r in self._commands.values()
                        if (r["status"] is None or ui_state(r["status"]) == "executing")
                        and now - r.get("read", -math.inf) >= self._feedback_period
                    ]
                for record in records:
                    self._poll(session, record)
                self._follow_previews(session)
            if self.phase == RUNNING and self._demo is not None and due:
                if next(self._demo, StopIteration) is StopIteration:
                    self._demo = None
            if self._finish_reason is not None and self.phase == RUNNING:
                self._end("ended" if self._finish_reason.startswith("Ended") else "interrupted", self._finish_reason)
                session.finish(self._finish_reason)
        except EpisodeEnded as ended:  # The budget, the runtime or a finish ended it.
            if self.phase != ENDED:
                self._end(str(ended) or "ended", str(ended))
        return self.phase != ENDED

    def track(self, client: str, skill: str, command_id: int, inputs: dict | None = None, session: Any = None) -> str:
        """Follow a command sent through ``session.<client>``; returns its UI id. Its status changes become events."""
        if client not in CHANNELS:
            raise ValueError(f"Unknown client {client!r}")
        with self._lock:
            uid = f"c{self._next_command}"
            self._next_command += 1
            record = dict(
                id=uid,
                client=client,
                channel=CHANNELS[client],
                skill=skill,
                command_id=command_id,
                inputs=plain(inputs or {}),
                started=self._time,
                status=None,
            )
            self._commands[uid] = record
        if session is not None:
            self._poll(session, record)
        return uid

    def _handle_requests(self, session: Any) -> None:
        for request in self._bridge.take_requests():
            kind = request.get("type")
            if kind == "estop":
                self._estop(session)
                continue
            try:
                if kind == "episode":
                    self._episode_request(session, request.get("action"))
                    continue
                if self.phase != RUNNING:
                    raise RuntimeError(
                        "The episode has not started; press Start." if self.phase == READY else "The episode has ended."
                    )
                if kind == "preview":
                    self._preview(session, request.get("skill"), request.get("inputs") or {})
                elif kind == "run":
                    self._run(session, request.get("skill"), request.get("inputs") or {})
                elif kind == "drive":
                    self._drive_request(session, request)
                elif kind == "cancel":
                    self._cancel(session, request.get("id"))
                else:
                    raise ValueError(f"Unknown request {kind!r}")
            except EpisodeEnded:
                raise
            except Exception as error:  # The API's own rejection, shown raw; the episode goes on.
                self._reject(request, error)

    def _estop(self, session: Any) -> None:
        """Stop the base, cancel the arm's commands and end the episode (on this pass)."""
        if self.phase == RUNNING:
            with self._lock:
                arm = [
                    r["command_id"]
                    for r in self._commands.values()
                    if r["client"] == "manipulation" and ui_state(r["status"] or "") == "executing"
                ]
            steps = [("stop the base", lambda: session.locomotion.stop())]
            steps += [("cancel an arm command", lambda c=c: session.manipulation.cancel(c)) for c in arm]
            for label, step in steps:  # Each on its own: one failing must not keep the episode running.
                try:
                    step()
                except EpisodeEnded:
                    break
                except Exception as error:
                    self._bridge.push_event(dict(type="error", message=f"E-stop could not {label}: {error}"))
        if self.phase != ENDED:
            self._finish_reason = "E-stop from the UI"
            if self.phase == READY:  # Nothing ran: end now.
                self._end("interrupted", self._finish_reason)
                session.finish(self._finish_reason)

    def _episode_request(self, session: Any, action: str) -> None:
        """Start (from ready) or End (while running) the episode."""
        if action == "start" and self.phase == READY:
            self._time = float(session.time())  # The first call: the World's clock starts.
            self.phase, self._start_wall = RUNNING, self._clock()
            self._bridge.push_event(dict(type="episode", sim_time=self._time, phase=RUNNING, status="running"))
            if self._demo_script is not None:
                self._demo = iter(self._demo_script(self, session))
        elif action == "end" and self.phase == RUNNING:
            self._finish_reason = "Ended from the UI"
        else:
            raise ValueError(f"Cannot {action!r} the episode while it is {self.phase}.")

    def _end(self, outcome: str, reason: str) -> None:
        """The episode is over: tell the page (the World's report follows with :meth:`show_report`)."""
        with self._lock:
            self.phase, self._end_wall, self._outcome = ENDED, self._clock(), (outcome, reason)
        self._bridge.push_event(
            dict(
                type="episode",
                sim_time=self._time,
                phase=ENDED,
                status=outcome,
                outcome=outcome,
                reason=reason,
                report=self.summary(),
            )
        )

    def summary(self) -> dict:
        """The page's side of the episode: outcome, times [s], commands per skill and failures."""
        start = self._start_wall
        end = self._end_wall if self._end_wall is not None else self._clock()
        outcome, reason = self._outcome or (self.phase, "")
        with self._lock:
            return dict(
                episode_id=self._episode_id,
                outcome=outcome,
                reason=reason,
                sim_seconds=self._time,
                wall_seconds=None if start is None else end - start,
                commands=json.loads(json.dumps(self._counts)),
                failures=json.loads(json.dumps(self._failures)),
                failure_list=list(self._failure_list),
                drive_seconds=self._drive_seconds,
            )

    def show_report(self, report: dict) -> dict:
        """After the World ends: the page's result card from its report (evaluation, times) and
        this summary; returns what was shown."""
        summary = self.summary()
        shown = dict(
            summary,
            sim_seconds=report.get("sim_seconds", summary["sim_seconds"]),
            evaluation=report.get("evaluation"),
            ended_by=report.get("ended_by"),
        )
        outcome, reason = self._outcome or (report.get("ended_by", ENDED), report.get("reason", ""))
        with self._lock:
            self.phase = ENDED
        self._bridge.push_event(
            dict(
                type="episode",
                sim_time=self._time,
                phase=ENDED,
                status=outcome,
                outcome=outcome,
                reason=reason,
                report=shown,
            )
        )
        return shown

    def _count(self, event: dict) -> None:
        """Count a command once it has ended: per skill and state, failures per code, and Drive time [s]."""
        with self._lock:
            if event["state"] == "executing" or event["id"] in self._counted:
                return
            self._counted.add(event["id"])
            skill, state = event["skill"], event["state"]
            self._counts.setdefault(skill, {}).setdefault(state, 0)
            self._counts[skill][state] += 1
            if skill == "drive":
                self._drive_seconds += max(0.0, event["sim_time"] - event["started"])
            if state == "failed":
                feedback = event.get("feedback") or {}
                code = feedback.get("failure") or feedback.get("status") or "failed"
                self._failures.setdefault(skill, {}).setdefault(code, 0)
                self._failures[skill][code] += 1
                self._failure_list.append(
                    dict(sim_time=event["sim_time"], skill=skill, code=code, reason=feedback.get("reason", ""))
                )

    def _held(self, session: Any) -> HeldObject | None:
        """The object the gripper holds, as the person cannot say: the configured box between the jaws."""
        if self._held_size is None or not session.manipulation.holding:
            return None
        return HeldObject(self._held_size, self._hand)

    def _run(self, session: Any, skill: str, inputs: dict) -> None:
        if "preview" in inputs:
            self._run_preview(session, skill, inputs["preview"])
            return
        if skill == "navigate_to":
            heading = inputs.get("heading")
            call = dict(
                x=float(inputs["x"]),
                y=float(inputs["y"]),
                heading=None if heading is None else float(heading),
                position_tolerance=float(inputs.get("position_tolerance", NavigationGoal.position_tolerance)),
                heading_tolerance=float(inputs.get("heading_tolerance", NavigationGoal.heading_tolerance)),
            )
            command = session.navigation.navigate_to(
                call["x"],
                call["y"],
                call["heading"],
                position_tolerance=call["position_tolerance"],
                heading_tolerance=call["heading_tolerance"],
            )
            self.track("navigation", skill, command, call, session)
        elif skill == "stop":
            self.track("locomotion", skill, session.locomotion.stop(), session=session)
        elif skill == "move_joints":
            positions = {name: float(inputs["positions"][name]) for name in self._arm_joints}
            duration = inputs.get("duration")
            manipulation = session.manipulation
            duration = manipulation.minimum_move_duration(positions) if duration is None else float(duration)
            command = manipulation.move_joints(positions, duration=duration)
            self.track("manipulation", skill, command, dict(positions=positions, duration=duration), session)
        elif skill == "execute_trajectory":
            self._run_path(session, inputs["waypoints"])
        elif skill in ("open_gripper", "close_gripper"):
            command = getattr(session.manipulation, skill)(duration=float(inputs.get("duration", 2.0)))
            self.track("manipulation", skill, command, session=session)
        elif skill == "stow":
            self.track("manipulation", skill, session.manipulation.stow(held=self._held(session)), session=session)
        elif skill == "unstow":
            self.track("manipulation", skill, session.manipulation.unstow(), session=session)
        elif skill in ("look_at", "move_to_pose", "move_along"):
            self._run_arm_skill(session, skill, inputs)
        else:
            raise ValueError(f"The UI cannot run {skill!r}")

    @staticmethod
    def _robot_pose(session: Any) -> tuple[float, float, float]:
        """The robot's x, y [m] and heading [rad] now, from the agent's own observation."""
        localization = session.observe().localization
        if localization is None or not localization.valid:
            raise RuntimeError("The robot's pose is unknown.")
        matrix = localization.pose.matrix.cpu().numpy()
        return float(matrix[0, 3]), float(matrix[1, 3]), float(math.atan2(matrix[1, 0], matrix[0, 0]))

    def _run_arm_skill(self, session: Any, skill: str, inputs: dict) -> None:
        """Skills with a clicked target or a direction, translated into the API's own arguments."""
        manipulation = session.manipulation
        packet = session.observe()
        yaw = self._yaw(packet)
        if skill == "move_along":
            axis, distance = translate.axis(inputs["direction"], yaw), float(inputs["distance_m"])
            times, poses = translate.line(tuple(manipulation.tool_pose("wrist")), axis, distance)
            call = dict(axis=axis, distance_m=distance, poses=poses)
            line = EndEffectorTrajectory(times, poses, packet.stamp.episode_id, tool="wrist")
            command = manipulation.execute_trajectory(line)
        else:
            target = inputs["target"]
            point = translate.point_target(target)
            call = dict(target=dict(point=list(point.position), view=target.get("view"), pixel=target.get("pixel")))
            if skill == "look_at":
                call.update(distance_m=float(inputs["distance_m"]), elevation_deg=float(inputs["elevation_deg"]))
                command = manipulation.look_at(
                    point, distance_m=call["distance_m"], elevation_deg=call["elevation_deg"], held=self._held(session)
                )
            else:
                call["pose"] = self._wrist_pose(target, inputs, yaw)
                command = manipulation.move_to_pose(call["pose"], tool="wrist", held=self._held(session))
        self.track("manipulation", skill, command, call, session)

    @staticmethod
    def _yaw(packet: Any) -> float:
        localization = packet.localization
        if localization is None or not localization.valid:
            raise RuntimeError("The robot's pose is unknown.")
        matrix = localization.pose.matrix.cpu().numpy()
        return float(math.atan2(matrix[1, 0], matrix[0, 0]))

    def _wrist_pose(self, target: dict, inputs: dict, yaw: float) -> tuple[float, ...]:
        """Move to pose's wrist pose [m, xyzw] for a resolved click and the page's choices."""
        return translate.wrist_pose(
            translate.point_target(target).position,
            inputs["orientation"],
            yaw=yaw,
            standoff_m=float(inputs["standoff_m"]),
            grasp=self._grasp,
            normal=target.get("normal"),
        )

    # Plan preview.
    def _preview(self, session: Any, skill: str, inputs: dict) -> None:
        """Plan Arm joints or Move to pose without moving; :meth:`_follow_previews` reports the plan."""
        if self._forward is None:
            raise RuntimeError("Plan preview needs the robot's kinematic model.")
        held = self._held(session)
        if skill == "move_joints":
            plan = session.manipulation.plan(
                goal={name: float(inputs["positions"][name]) for name in self._arm_joints}, held=held
            )
        elif skill == "move_to_pose":
            pose = self._wrist_pose(inputs["target"], inputs, self._robot_pose(session)[2])
            plan = session.manipulation.plan(pose_xyzw=pose, tool="wrist", held=held)
        else:
            raise ValueError(f"The UI cannot preview {skill!r}.")
        with self._lock:
            preview_id = f"p{len(self._previews) + 1}"
            self._previews[preview_id] = dict(skill=skill, inputs=plain(inputs), plan=plan, trajectory=None)
        self._bridge.push_event(
            dict(type="preview", id=preview_id, skill=skill, sim_time=self._time, status="planning")
        )

    def _follow_previews(self, session: Any) -> None:
        """Report each finished preview plan once: the gripper path and the final arm, or the raw failure."""
        with self._lock:
            done = [(i, p) for i, p in self._previews.items() if p["plan"] is not None and p["plan"].done()]
        for preview_id, preview in done:
            plan, preview["plan"] = preview["plan"], None
            event = dict(type="preview", id=preview_id, skill=preview["skill"], sim_time=self._time)
            try:
                trajectory = plan.result()
            except Exception as error:  # The planner's own failure, shown raw.
                event.update(status="failed", reason=f"{type(error).__name__}: {error}")
            else:
                preview["trajectory"] = trajectory
                event.update(status="ready", **self._plan_drawing(session, trajectory))
            self._bridge.push_event(event)

    def _plan_drawing(self, session: Any, trajectory: JointTrajectory) -> dict:
        """A plan as the page draws it: the grasp point's path and the arm at the end, in the body
        frame [m]; the path in the world frame [m] for the map; its duration [s]."""
        packet = session.observe()
        measured = dict(zip(packet.joints.names, packet.joints.positions.cpu().tolist(), strict=True))
        rows = np.linspace(0, len(trajectory.times) - 1, min(PREVIEW_SAMPLES, len(trajectory.times))).round()
        grasp = self._grasp[:, 3]
        path = []
        for row in rows.astype(int):
            arm_joints = dict(zip(trajectory.names, trajectory.positions[row].tolist(), strict=True))
            links = self._forward({**measured, **arm_joints})
            path.append((links[self._wrist_link] @ grasp)[:3])
        arm = [links[name][:3, 3] for name in self._arm_links if name in links] + [path[-1]]
        matrix = packet.localization.pose.matrix.cpu().numpy()
        x, y, yaw = float(matrix[0, 3]), float(matrix[1, 3]), math.atan2(matrix[1, 0], matrix[0, 0])
        turn = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        world = [(turn @ point[:2] + [x, y]).tolist() for point in path]
        return dict(
            duration=float(trajectory.times[-1]),
            path_body=[point.tolist() for point in path],
            arm_body=[point.tolist() for point in arm],
            path_world=world,
        )

    def _run_preview(self, session: Any, skill: str, preview_id: str) -> None:
        """Run a previewed plan as it was shown: ``manipulation.execute_trajectory`` of that trajectory."""
        with self._lock:
            preview = self._previews.get(preview_id)
        if preview is None or preview["trajectory"] is None:
            raise ValueError(f"No ready plan {preview_id!r} to run; preview again.")
        command = session.manipulation.execute_trajectory(preview["trajectory"])
        self.track("manipulation", skill, command, dict(preview=preview_id, previewed=preview["inputs"]), session)

    def _run_path(self, session: Any, waypoints: list[dict]) -> None:
        """Saved arm poses [rad] after the given seconds each, from the measured pose, as one trajectory."""
        packet = session.observe()
        joints = packet.joints
        if joints is None or not joints.valid:
            raise RuntimeError("No valid joint measurement to start the path from.")
        measured = dict(zip(joints.names, joints.positions.cpu().tolist(), strict=True))
        times, positions = [0.0], [[measured[name] for name in self._arm_joints]]
        for waypoint in waypoints:
            times.append(times[-1] + float(waypoint["seconds"]))
            positions.append([float(waypoint["positions"][name]) for name in self._arm_joints])
        trajectory = JointTrajectory(self._arm_joints, np.array(times), np.array(positions), packet.stamp.episode_id)
        command = session.manipulation.execute_trajectory(trajectory)
        self.track("manipulation", "execute_trajectory", command, dict(waypoints=waypoints, times=times), session)

    def _drive_request(self, session: Any, request: dict) -> None:
        """Renew one Drive command while the person holds a key; each renewal lasts 0.5 s."""
        speed, turn = self._drive_limits["max_speed"], self._drive_limits["max_yaw_speed"]
        velocity = [float(request.get(name, 0.0)) for name in ("v_x", "v_y", "v_rot")]
        if not all(map(math.isfinite, velocity)):
            raise ValueError("Drive velocity must be finite.")
        velocity = [
            max(-speed, min(speed, velocity[0])),
            max(-speed, min(speed, velocity[1])),
            max(-turn, min(turn, velocity[2])),
        ]
        command = session.locomotion.command_velocity(*velocity, duration=0.5)
        with self._lock:
            record = self._drive
            if record is not None and ui_state(record["status"]) == "executing":
                record["command_id"] = command  # The same Drive, renewed; the old command id is overridden.
                return
        uid = self.track("locomotion", "drive", command, dict(zip(("v_x", "v_y", "v_rot"), velocity)), session)
        with self._lock:
            self._drive = self._commands[uid]

    def _cancel(self, session: Any, uid: str) -> None:
        with self._lock:
            record = self._commands.get(uid)
        if record is None:
            raise ValueError(f"No command {uid!r}")
        if record["client"] == "locomotion":
            self.track("locomotion", "stop", session.locomotion.stop(), session=session)
        else:
            getattr(session, record["client"]).cancel(record["command_id"])

    def _reject(self, request: dict, error: Exception) -> None:
        """A request the API refused: a failed command carrying the raw error."""
        kind = request.get("type")
        skill = request.get("skill") or kind
        channel = "task" if kind == "episode" else "base" if skill in BASE_REQUESTS else "arm"
        inputs = request.get("inputs") or {}
        with self._lock:
            uid = f"c{self._next_command}"
            self._next_command += 1
        now = self._time
        event = dict(
            type="command",
            id=uid,
            channel=channel,
            skill=skill,
            inputs=plain(inputs),
            started=now,
            sim_time=now,
            status="rejected",
            state="failed",
            feedback=dict(status="rejected", reason=f"{type(error).__name__}: {error}"),
        )
        self._bridge.push_event(event)
        self._count(event)

    def _poll(self, session: Any, record: dict) -> None:
        feedback = plain(getattr(session, record["client"]).feedback(record["command_id"]))
        with self._lock:
            record["read"] = self._clock()  # Read again after feedback_period.
            record["feedback"] = feedback  # The latest, for the telemetry drawer.
            status = feedback.get("status", "") if isinstance(feedback, dict) else str(feedback)
            if status == record["status"]:
                return
            record["status"] = status
            event = {key: record[key] for key in ("id", "channel", "skill", "inputs", "started")}
        event.update(type="command", sim_time=self._time, status=status, state=ui_state(status), feedback=feedback)
        self._bridge.push_event(event)
        self._count(event)

    # ------------------------------------------------------------------------------------------
    # Viewer side (main thread): what the person sees, read-only.
    def update(self, view: Any) -> None:
        """The World's viewer: called on the main thread once per camera period."""
        self._time, self._episode_id = float(view.time_seconds), view.episode_id
        packet = view.observe()
        self._publish_frames(packet)
        self._publish_steps(view)
        self._publish_map(view)
        self._publish_state(view, packet)

    @property
    def map_updates(self) -> int:
        """How many times the navigation costmap changed on the page."""
        return self._map_version or 0

    def _publish_frames(self, packet: Any) -> None:
        for name in self.cameras:
            camera = packet.cameras.get(name)
            if camera is None or not camera.valid or camera.rgb is None or camera.depth is None:
                continue
            if self._last_ticks.get(name) == camera.stamp.tick:
                continue
            self._last_ticks[name] = camera.stamp.tick
            times = self._camera_times.setdefault(name, collections.deque())
            times.append(float(camera.stamp.time_seconds))
            while times[-1] - times[0] > CAMERA_WINDOW_S:
                times.popleft()
            self._bridge.publish_frame(camera_frame(packet, name))

    def _publish_steps(self, view: Any) -> None:
        lines = view.narration(self._steps_seen)
        self._steps_seen += len(lines)
        for time_seconds, kind, text in lines:
            if kind in STEP_KINDS:
                self._bridge.push_event(dict(type="step", sim_time=time_seconds, kind=kind, text=text))

    def _publish_map(self, view: Any) -> None:
        """Navigation's costmap as the page's map: obstacles, unknown, and where the robot's centre may not go."""
        if self._clock() - self._last_map_wall < self._map_period:
            return
        self._last_map_wall = self._clock()
        try:
            costmap = view.costmap()
        except (RuntimeError, ValueError, TimeoutError) as error:  # Navigation reports it too, when asked to plan.
            if str(error) != self._map_error:
                self._map_error = str(error)
                self._bridge.push_event(dict(type="error", message=f"Navigation map: {error}"))
            return
        self._map_error = None
        if costmap is None:
            return
        costs, origin, resolution = costmap
        if self._last_costs is not None and np.array_equal(costs, self._last_costs):
            return
        self._last_costs = costs
        self._map_version = (self._map_version or 0) + 1
        self._bridge.publish_map(
            MapImage(
                cells=np.where(costs == 255, -1, np.where(costs >= 254, 100, 0)).astype(np.int8),
                origin=(float(origin[0]), float(origin[1])),
                resolution=float(resolution),
                version=self._map_version,
                blocked=(costs >= 253) & (costs != 255),
            )
        )

    def _publish_state(self, view: Any, packet: Any) -> None:
        wall, sim = self._clock() - self._started, self._time
        self._timing.append((wall, sim))
        while self._timing and wall - self._timing[0][0] > 5.0:
            self._timing.popleft()
        first_wall, first_sim = self._timing[0]
        rtf = (sim - first_sim) / (wall - first_wall) if wall - first_wall > 0.5 else None
        pose = speed = None
        if packet.localization is not None and packet.localization.valid:
            matrix = packet.localization.pose.matrix.cpu().numpy()
            pose = dict(
                x=float(matrix[0, 3]),
                y=float(matrix[1, 3]),
                z=float(matrix[2, 3]),
                yaw=float(math.atan2(matrix[1, 0], matrix[0, 0])),
            )
            speed = float(np.linalg.norm(packet.localization.linear_velocity.cpu().numpy()[:2]))
        arm = None
        if packet.joints is not None and packet.joints.valid:
            positions = dict(zip(packet.joints.names, packet.joints.positions.cpu().tolist(), strict=True))
            joints = {name: positions[name] for name in self._arm_joints if name in positions}
            stowed = None
            if len(joints) == len(self._arm_joints):
                stowed = (
                    max(abs(joints[n] - q) for n, q in zip(self._arm_joints, self._stow, strict=True))
                    <= self._stow_tolerance
                )
            arm = dict(joints=joints, gripper=positions.get(self._gripper_joint), stowed=stowed)
        telemetry = self._telemetry(packet, sim)
        self._write_telemetry(dict(sim_time=sim, wall_time=wall, rtf=rtf, pose=pose, arm=arm, **telemetry))
        with self._lock:
            counts, failures, phase = (
                json.loads(json.dumps(self._counts)),
                json.loads(json.dumps(self._failures)),
                self.phase,
            )
        self._bridge.publish_state(
            dict(
                type="state",
                sim_time=sim,
                wall_time=wall,
                rtf=rtf,
                episode_id=view.episode_id,
                status=view.status,
                pose=pose,
                speed=speed,
                arm=arm,
                holding=bool(view.holding),
                phase=phase,
                task=None
                if not self._task
                else dict(**self._task, budget=self._budget, elapsed=sim if phase != READY else 0.0),
                telemetry=dict(**telemetry, commands=counts, failures=failures),
            )
        )

    def _telemetry(self, packet: Any, sim: float) -> dict:
        """What the telemetry drawer plots, from the viewer's observation: base velocity [m/s, body
        frame] and turn rate [rad/s], arm joint speeds [rad/s], hand position [m, world], the running
        arm command's latest feedback, each camera's frame rate [Hz] and age [s], and the runner's health."""
        base = None
        matrix = None
        if packet.localization is not None and packet.localization.valid:
            matrix = packet.localization.pose.matrix.cpu().numpy()
            yaw = math.atan2(matrix[1, 0], matrix[0, 0])
            rate = None
            if self._last_yaw is not None and sim > self._last_yaw[0]:
                turned = (yaw - self._last_yaw[1] + math.pi) % (2 * math.pi) - math.pi
                rate = turned / (sim - self._last_yaw[0])
            self._last_yaw = (sim, yaw)
            velocity = packet.localization.linear_velocity.cpu().numpy()
            base = dict(vx=float(velocity[0]), vy=float(velocity[1]), yaw_rate=rate)
        joint_speeds = hand = None
        if packet.joints is not None and packet.joints.valid:
            speeds = dict(zip(packet.joints.names, packet.joints.velocities.cpu().tolist(), strict=True))
            joint_speeds = {name: speeds[name] for name in (*self._arm_joints, self._gripper_joint) if name in speeds}
            if self._forward is not None and matrix is not None:
                positions = dict(zip(packet.joints.names, packet.joints.positions.cpu().tolist(), strict=True))
                wrist = self._forward(positions).get(self._wrist_link)
                if wrist is not None:
                    hand = (matrix @ wrist @ self._hand)[:3, 3].tolist()
        tracking = None
        with self._lock:
            for record in self._commands.values():
                if record["channel"] == "arm" and ui_state(record["status"] or "") == "executing":
                    feedback = record.get("feedback") or {}
                    tracking = dict(
                        id=record["id"],
                        skill=record["skill"],
                        **{key: feedback.get(key) for key in ("joint_error", "pose_error", "progress")},
                    )
        cameras = {}
        for name, times in self._camera_times.items():
            span = times[-1] - times[0]
            cameras[name] = dict(fps=(len(times) - 1) / span if span > 0 else None, age=sim - times[-1])
        return dict(
            base=base,
            joint_speeds=joint_speeds,
            hand=hand,
            tracking=tracking,
            cameras=cameras,
            health=None if self._health is None else plain(self._health()),
        )

    def _write_telemetry(self, line: dict) -> None:
        """Append one line to the telemetry file every telemetry period of sim time."""
        if self._telemetry_file is None or line["sim_time"] - self._telemetry_written < self._telemetry_period:
            return
        self._telemetry_written = line["sim_time"]
        self._telemetry_file.write(json.dumps(plain(line), separators=(",", ":")) + "\n")
        self._telemetry_file.flush()

    def close(self) -> None:
        """Close the telemetry file."""
        if self._telemetry_file is not None:
            self._telemetry_file.close()
            self._telemetry_file = None
