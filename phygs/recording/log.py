# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An episode's recording (``episode.rrd``): every time series of the run, written once, by the part
that owns it, through the Rerun SDK. The viewer (``rerun episode.rrd``, native or in the browser),
the replay and analysis scripts read it; the scored result stays in ``report.json``.

Every row carries four timelines:

- ``tick``: the physics step since the episode's start (an integer);
- ``sim_time``: simulated seconds since the start;
- ``wall_time``: the host clock (the charged clock stops simulated time while the agent thinks, so
  wall time is the only one that shows thinking);
- ``seq``: the order of rows within one instant.

Entities, by branch (``/truth`` is what is true, kept apart from what the agent could know, so a
public copy is the recording without it):

- ``/truth/<robot>``: the robot's URDF, logged once: its collision geometry (what the physics
  simulates; the visual meshes are 45 times larger) and its fixed joints (``joints/<joint>``). Every
  state sample (each camera period): ``base`` (its root link in the world), ``joints/<joint>`` (each
  moving joint's transform) and ``joint_positions`` [rad or m], one plot line per joint.
- ``/truth/scene/static``: colliders fixed to the world (boxes, once).
- ``/truth/scene/bodies/<body path>``: each scene body's pose, logged at the first sample and then
  when it moved more than 1 mm or 0.25 degrees; its colliders below it (boxes in its frame, once).
- ``/truth/contacts``: arm contacts above the record threshold, when a link starts touching a body.
- ``/estimate/holding``: whether the robot's gripper holds an object (1) or not (0), every sample.
- ``/control/base_command``: the base's commanded body velocity [m/s, m/s, rad/s], every sample (NaN:
  none); ``/control/commands``: the command timeline's spans, once at the end.
- ``/nav/route``: Nav2's route (a line at the floor, world frame) and goal, when it changed;
  ``/nav/costmap``: Nav2's global costmap (an image of costs 0-255, row 0 at the lowest y) with its
  origin and cell size, when it changed.
- ``/planner/plan``: each finished arm plan's display (:class:`~phygs.recording.panels.PlannerView`)
  in the robot's body frame: its label, voxel size and floor, and below it ``voxels``, ``boxes``,
  ``arm_start``, ``arm_path``, ``arm_end``, ``end_spheres`` and ``tool_path``.
- ``/agent/calls``: every agent call and step marker, as the audit writes it.
- ``/events/phase``, ``/events/anomaly``: the episode's phases (settling included) and anomalies.
- ``/perf/*``: real-time factor, tick time and machine load, once per wall-clock second.

Recording properties (``rerun`` shows them beside the data): the run id, :data:`SCHEMA`, the episode
spec, the provenance, ``layout`` (the robot's entity name, the joint and body order) and ``replay``
(what ``phygs replay`` needs to build the scene again). :mod:`.reader` reads a recording back.

Logging never fails or changes the episode: every call is guarded, and the first error closes the
recording and is kept in :attr:`EpisodeLog.status`, which the report records. The file is a stream,
made safe on disk at every phase: a run that dies half-way still reads back up to the last chunk
written (:mod:`.reader` repairs the file's missing index). The recording stream is this
object's own, never Rerun's global one, so several Worlds can record in one process.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

SCHEMA = 1
"""The entity layout and meaning above; a reader checks it (the ``phygs_log_schema`` property)."""
FILE = "episode.rrd"
LEVELS = ("full", "report_only")
POSITION_STEP = 0.001  # [m]; a body that moved this far since its last logged pose is logged again.
ROTATION_STEP = math.radians(0.25)  # [rad]; likewise for its rotation.
PERF_PERIOD = 1.0  # [s wall]; the /perf rows.
WORLD_FRAME = "tf#/truth"  # The frame the robot's base and the scene are placed in.
ROUTE_HEIGHT = 0.05  # [m]; Nav2's route is drawn this high above the floor.


def new_run_id() -> str:
    """A run's id: it names the recording and joins the run's files (report, result, labels)."""
    return uuid.uuid4().hex


def _guarded(method: Callable) -> Callable:
    """Run a logging call; on its first error close the recording and keep the error, never raise.
    Rerun's own checks raise inside it (strict mode, for this thread and call only) rather than warn,
    so no row is dropped unnoticed."""

    @functools.wraps(method)
    def call(self: EpisodeLog, *args, **kwargs):
        if self._stream is None:
            return None
        try:
            with self._rr.error_utils.catch_and_log_exceptions(method.__name__, strict=True):
                return method(self, *args, **kwargs)
        except Exception as error:  # noqa: BLE001 - logging must not fail the episode (module doc).
            self._fail(f"{method.__name__}: {type(error).__name__}: {error}")
            return None

    return call


class EpisodeLog:
    """One episode's recording at ``path`` (a ``.rrd`` file); ``live`` also streams it to a viewer at
    that gRPC address (``rerun+http://127.0.0.1:9876/proxy``: started by the person who watches, so
    nothing here listens on the network). ``level`` ``report_only`` writes no recording."""

    def __init__(self, path: Path, *, run_id: str, level: str = "full", live: str | None = None) -> None:
        if level not in LEVELS:
            raise ValueError(f"Unknown recording level {level!r}; expected one of {LEVELS}.")
        self.path, self.run_id, self.level = Path(path), run_id, level
        self.status, self.error = ("recording", None) if level != "report_only" else ("off", None)
        self._stream = None
        self._tick, self._sim_time = 0, 0.0
        self._seq = itertools.count()  # next() is atomic: the agent's thread logs its calls too.
        self._urdf_joints: dict[str, _Joint] = {}
        self._robot = self._root = None
        self._joint_buffer: list[tuple] = []  # (tick, sim time, base pose, joints, holding, command)
        self._joint_names: tuple[str, ...] = ()
        self._bodies: tuple[str, ...] = ()
        self._body_poses: np.ndarray | None = None
        self._route: tuple | None = None  # The last route given, and its bytes (NaN goals compare equal).
        self._costmap: tuple | None = None  # The last costmap logged: (costs, origin, cell).
        self._plan = None  # The last plan display logged.
        self._perf_wall, self._perf_sim, self._perf_ticks = None, 0.0, 0
        if level == "report_only":
            return
        try:
            import rerun as rr

            stream = rr.RecordingStream("phygs", recording_id=run_id)
            stream.set_sinks(*[rr.FileSink(str(self.path))] + ([rr.GrpcSink(live)] if live else []))
        except Exception as error:  # noqa: BLE001 - see _guarded.
            self.status, self.error = "failed", f"open: {type(error).__name__}: {error}"
            return
        self._rr, self._stream = rr, stream
        self.properties(phygs_log_schema=SCHEMA, run_id=run_id)
        self._series_names()

    @property
    def robot(self) -> str | None:
        """The robot's URDF name (its ``/truth/<robot>`` entities), once :meth:`robot_model` ran."""
        return self._robot

    @property
    def root(self) -> str | None:
        """The robot's root link (the base's frame), once :meth:`robot_model` ran."""
        return self._root

    # --- Time ---

    def at(self, tick: int, sim_time: float) -> None:
        """Set the episode's tick and simulated time [s] for the rows that follow."""
        self._tick, self._sim_time = int(tick), float(sim_time)

    def _stamp(self) -> None:
        stream = self._stream
        stream.set_time("tick", sequence=self._tick)
        stream.set_time("sim_time", duration=self._sim_time)
        stream.set_time("wall_time", timestamp=time.time())
        stream.set_time("seq", sequence=next(self._seq))

    # --- The recording's description ---

    @_guarded
    def properties(self, **values) -> None:
        """Recording properties, each a JSON-able value, stored as JSON text."""
        for name, value in values.items():
            self._stream.send_property(name, self._rr.AnyValues(value=json.dumps(value, default=str)))

    @_guarded
    def _series_names(self) -> None:
        self._stream.log("/control/base_command", self._rr.SeriesLines(names=["forward", "left", "turn"]), static=True)

    @_guarded
    def blueprint(self, blueprint) -> None:
        """The viewer's layout (:mod:`phygs.recording.blueprint`)."""
        self._stream.send_blueprint(blueprint)

    # --- /truth: the robot ---

    @_guarded
    def robot_model(self, urdf: Path, joint_names: Sequence[str]) -> None:
        """The robot's URDF, once (its collision geometry and fixed joints), and the joints whose
        positions :meth:`robot_state` will log (in this order; joints the URDF lacks are plotted but
        not drawn)."""
        rr, stream = self._rr, self._stream
        tree = rr.urdf.UrdfTree.from_file_path(str(urdf), "truth")
        self._robot, self._root = tree.name, tree.root_link().name
        # Rerun's own URDF logging also fixes every moving joint at zero (static), which would hide
        # the joints robot_state moves: the geometry here, the joints' transforms below.
        geometry = tree.stream(include_joint_transforms=False).to_chunks()
        stream.send_chunks(chunk for chunk in geometry if "visual_geometries" not in str(chunk.entity_path))
        for joint in tree.joints():
            if joint.joint_type == "fixed":
                transform = rr.Transform3D(
                    translation=joint.origin_xyz,
                    quaternion=_quaternion_from_rpy(*joint.origin_rpy),
                    parent_frame=joint.parent_link,
                    child_frame=joint.child_link,
                )
                stream.log(_path("truth", self._robot, "joints", joint.name), transform, static=True)
        self._joint_names = tuple(joint_names)
        stream.log(_path("truth", self._robot, "joint_positions"), rr.SeriesLines(names=self._joint_names), static=True)
        moving = {joint.name: joint for joint in tree.joints() if joint.joint_type != "fixed"}
        self._urdf_joints = {name: _Joint.of(moving[name]) for name in self._joint_names if name in moving}
        self.properties(robot_layout=dict(entity=self._robot, root=self._root, joints=list(self._joint_names)))

    def robot_state(
        self,
        base_pose: np.ndarray,
        joint_positions: np.ndarray,
        *,
        holding: bool = False,
        command: Sequence[float] | None = None,
    ) -> None:
        """The robot now: its root link's world pose [x, y, z, qx, qy, qz, qw], the joint positions in
        :meth:`robot_model`'s order, whether the gripper holds an object and the base's commanded
        velocity [m/s, m/s, rad/s] (None: none). Buffered and sent once per simulated second."""
        if self._stream is None or self._robot is None:
            return
        command = np.full(3, np.nan) if command is None else np.asarray(command, dtype=float).reshape(3)
        base, joints = np.array(base_pose, dtype=float), np.array(joint_positions, dtype=float)  # Copies.
        self._joint_buffer.append((self._tick, self._sim_time, base, joints, float(bool(holding)), command))
        if len(self._joint_buffer) > 1 and self._sim_time - self._joint_buffer[0][1] >= 1.0:
            self._send_robot_states()

    @_guarded
    def _send_robot_states(self) -> None:
        rr, stream, buffer = self._rr, self._stream, self._joint_buffer
        if not buffer:
            return
        self._joint_buffer = []
        ticks = np.array([row[0] for row in buffer])
        times = np.array([row[1] for row in buffer])
        bases = np.stack([row[2] for row in buffer])
        joints = np.stack([row[3] for row in buffer])
        count = len(buffer)
        indexes = [rr.TimeColumn("tick", sequence=ticks), rr.TimeColumn("sim_time", duration=times)]
        holding = np.array([row[4] for row in buffer])
        stream.send_columns("/estimate/holding", indexes=indexes, columns=rr.Scalars.columns(scalars=holding))
        commands = np.stack([row[5] for row in buffer])
        stream.send_columns("/control/base_command", indexes=indexes, columns=rr.Scalars.columns(scalars=commands))
        stream.send_columns(
            _path("truth", self._robot, "base"),
            indexes=indexes,
            columns=rr.Transform3D.columns(
                translation=bases[:, :3],
                quaternion=bases[:, 3:7],
                parent_frame=[WORLD_FRAME] * count,
                child_frame=[self._root] * count,
            ),
        )
        stream.send_columns(
            _path("truth", self._robot, "joint_positions"), indexes=indexes, columns=rr.Scalars.columns(scalars=joints)
        )
        for column, name in enumerate(self._joint_names):
            joint = self._urdf_joints.get(name)
            if joint is not None:
                translation, quaternion = joint.transforms(joints[:, column])
                stream.send_columns(
                    _path("truth", self._robot, "joints", name),
                    indexes=indexes,
                    columns=rr.Transform3D.columns(
                        translation=translation,
                        quaternion=quaternion,
                        parent_frame=[joint.parent] * count,
                        child_frame=[joint.child] * count,
                    ),
                )

    # --- /truth: the scene ---

    @_guarded
    def scene(self, colliders: Sequence[tuple[str | None, np.ndarray, np.ndarray]], bodies: Sequence[str]) -> None:
        """The scene's colliders, once: (body path or None for the world, center [m], size [m]),
        centers in the body's frame; and the scene bodies :meth:`body_poses` will log, in its order."""
        rr = self._rr
        self._bodies = tuple(bodies)
        self.properties(scene_bodies=list(self._bodies))
        fixed = [(center, size) for body, center, size in colliders if body is None]
        if fixed:
            self._stream.log(
                "/truth/scene/static",
                rr.Boxes3D(centers=[c for c, _ in fixed], sizes=[s for _, s in fixed], colors=[180, 180, 175]),
                static=True,
            )
        by_body: dict[str, list] = {}
        for body, center, size in colliders:
            if body is not None:
                by_body.setdefault(body, []).append((center, size))
        for body, boxes in by_body.items():
            self._stream.log(
                _path("truth", "scene", "bodies", body, "colliders"),
                rr.Boxes3D(centers=[c for c, _ in boxes], sizes=[s for _, s in boxes], colors=[120, 170, 220]),
                static=True,
            )

    @_guarded
    def body_poses(self, poses: np.ndarray) -> None:
        """Every scene body's world pose [B, 7] in :meth:`scene`'s order; logs the ones that moved
        (all of them the first time)."""
        poses = np.asarray(poses, dtype=np.float32)  # The simulation's precision: the moved test as the state log's.
        moved = np.ones(len(poses), dtype=bool) if self._body_poses is None else _moved(self._body_poses, poses)
        if not moved.any():
            return
        if self._body_poses is None:
            self._body_poses = poses.copy()
        self._body_poses[moved] = poses[moved]
        self._stamp()
        rr = self._rr
        for index in np.flatnonzero(moved):
            pose = poses[index]
            self._stream.log(
                _path("truth", "scene", "bodies", self._bodies[index]),
                rr.Transform3D(translation=pose[:3], quaternion=pose[3:7]),
            )

    @_guarded
    def contact(self, link: str, body: str, force: float, step: str) -> None:
        """A robot link started touching a scene body with ``force`` [N] during ``step``."""
        self._stamp()
        self._stream.log(
            "/truth/contacts", self._rr.TextLog(f"{link} touches {body}: {force:.1f} N ({step})", level="WARN")
        )

    # --- Navigation, the arm planner, the command timeline ---

    @_guarded
    def route(self, goal: tuple | None, points: Sequence[Sequence[float]]) -> None:
        """Nav2's route now: the goal (x, y [m], heading [rad] or None) and the points [P, 2] [m]
        (no goal and no points: none); logged when it differs from the last one."""
        if self._route is not None and goal == self._route[0] and points is self._route[1]:
            return  # The same route object as last time (Nav2's backend keeps it until it replans).
        goal = None if goal is None else [math.nan if value is None else float(value) for value in goal]
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        key = (np.asarray(goal if goal is not None else (), dtype=np.float32).tobytes(), points.tobytes())
        changed = self._route is None or key != self._route[2]
        self._route = (goal, points, key) if goal is None else (tuple(goal), points, key)
        if not changed:
            return
        rr = self._rr
        line = np.column_stack([points, np.full(len(points), ROUTE_HEIGHT, np.float32)])
        self._stamp()
        self._stream.log(
            "/nav/route",
            rr.LineStrips3D([line] if len(line) else [], colors=[40, 160, 60]),
            rr.AnyValues(goal=[math.nan] * 3 if goal is None else (goal + [math.nan] * 3)[:3]),
        )

    @_guarded
    def costmap(self, costs: np.ndarray, origin: Sequence[float], cell: float) -> None:
        """Nav2's global costmap (costs 0-255 [rows, cols], row 0 at the lowest y; origin [m]; cell
        size [m]); logged when it differs from the last one."""
        costs = np.asarray(costs, dtype=np.uint8)
        origin, cell = [float(origin[0]), float(origin[1])], float(cell)
        last = self._costmap
        if last is not None and last[1] == origin and last[2] == cell and np.array_equal(last[0], costs):
            return
        self._costmap = (costs.copy(), origin, cell)
        self._stamp()
        self._stream.log(
            "/nav/costmap", self._rr.Image(costs, color_model="L"), self._rr.AnyValues(origin=origin, cell=cell)
        )

    @_guarded
    def plan(self, view) -> None:
        """An arm plan's display (:class:`~phygs.recording.panels.PlannerView`, body frame), when it is
        a new one (the planner makes one per finished plan)."""
        if view is self._plan:
            return
        self._plan = view
        rr, stream = self._rr, self._stream
        self._stamp()
        floor = math.nan if view.floor is None else float(view.floor)
        stream.log("/planner/plan", rr.AnyValues(label=view.label, voxel_size=float(view.voxel_size), floor=floor))
        voxels = np.asarray(view.voxels, dtype=np.float32).reshape(-1, 3)
        stream.log("/planner/plan/voxels", rr.Points3D(voxels, radii=view.voxel_size / 2, colors=[200, 120, 60]))
        boxes = [(center, size) for center, size in view.boxes]
        stream.log(
            "/planner/plan/boxes",
            rr.Boxes3D(centers=[c for c, _ in boxes], sizes=[s for _, s in boxes], colors=[120, 170, 220]),
        )
        for name in ("arm_start", "arm_end", "tool_path"):
            line = np.asarray(getattr(view, name), dtype=np.float32).reshape(-1, 3)
            stream.log(f"/planner/plan/{name}", rr.LineStrips3D([line] if len(line) else []))
        frames = [np.asarray(frame, dtype=np.float32).reshape(-1, 3) for frame in view.arm_path]
        stream.log("/planner/plan/arm_path", rr.LineStrips3D(frames))
        spheres = np.asarray(view.end_spheres, dtype=np.float32).reshape(-1, 4)
        stream.log("/planner/plan/end_spheres", rr.Points3D(spheres[:, :3], radii=spheres[:, 3]))

    @_guarded
    def commands(self, spans: Sequence[dict]) -> None:
        """The command timeline's spans (lane, name, start, end [s] or None, status), once at the end."""
        self._stamp()
        self._stream.log("/control/commands", self._rr.AnyValues(spans=json.dumps(list(spans), default=str)))

    # --- Agent and episode events ---

    @_guarded
    def agent_call(self, line: dict) -> None:
        """One audit line (an agent call with its result, or a step marker); called from the agent's
        thread (Rerun keeps the time per thread, so each row is stamped where it is logged)."""
        self._stamp()
        call = str(line.get("call", ""))
        error = line.get("error")
        text = f"{call} {json.dumps(line.get('args', {}), default=str)}"
        if "result" in line or error is not None:
            text += f" -> {json.dumps(error if error is not None else line.get('result'), default=str)}"
        self._stream.log(
            "/agent/calls",
            self._rr.TextLog(text[:2000], level="ERROR" if error is not None else "INFO"),
            self._rr.AnyValues(line=json.dumps(line, default=str)),
        )

    @_guarded
    def phase(self, name: str, detail: str = "") -> None:
        """The episode entered a phase (settle, start_check, agent, end, stop, end_settle,
        evaluation); what was logged so far is made safe on disk."""
        self._stamp()
        self._stream.log("/events/phase", self._rr.TextLog(f"{name}{': ' + detail if detail else ''}", level="INFO"))
        self.flush()

    @_guarded
    def anomaly(self, text: str) -> None:
        """Something physical went wrong (a fall, a runtime failure)."""
        self._stamp()
        self._stream.log("/events/anomaly", self._rr.TextLog(text, level="ERROR"))

    # --- /perf ---

    @_guarded
    def tick_done(self, tick_seconds: float) -> None:
        """One physics tick took ``tick_seconds`` [s wall]; once per wall second logs the real-time
        factor, the mean tick time and the machine's load."""
        now = time.perf_counter()
        if self._perf_wall is None:
            self._perf_wall, self._perf_sim, self._perf_ticks, self._perf_busy = now, self._sim_time, 0, 0.0
            return
        self._perf_ticks += 1
        self._perf_busy += tick_seconds
        elapsed = now - self._perf_wall
        if elapsed < PERF_PERIOD:
            return
        rr = self._rr
        self._stamp()
        self._stream.log("/perf/rtf", rr.Scalars((self._sim_time - self._perf_sim) / elapsed))
        self._stream.log("/perf/tick_ms", rr.Scalars(1000.0 * self._perf_busy / max(1, self._perf_ticks)))
        self._stream.log("/perf/load_1min", rr.Scalars(os.getloadavg()[0]))
        self._perf_wall, self._perf_sim, self._perf_ticks, self._perf_busy = now, self._sim_time, 0, 0.0

    # --- End ---

    def close(self) -> dict:
        """Send what is buffered, close the file, and return the report's ``recording`` entry."""
        if self._stream is not None:
            self._send_robot_states()
        if self._stream is not None:
            try:
                self._stream.flush()
                self._stream.disconnect()
            except Exception as error:  # noqa: BLE001 - the report keeps it (module doc).
                self.status, self.error = "failed", f"close: {type(error).__name__}: {error}"
            else:
                self.status = "complete"
            self._stream = None
        return dict(
            status=self.status,
            file=self.path.name if self.level != "report_only" else None,
            error=self.error,
            schema=SCHEMA,
            run_id=self.run_id,
        )

    def flush(self) -> None:
        """Make what was logged so far safe on disk (each phase change; a crash keeps it)."""
        if self._stream is not None:
            self._send_robot_states()
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception as error:  # noqa: BLE001 - see _guarded.
                self._fail(f"flush: {type(error).__name__}: {error}")

    def _fail(self, error: str) -> None:
        stream, self._stream = self._stream, None
        if stream is None:  # Another thread failed or closed it first; its outcome is the one kept.
            return
        self.status, self.error = "failed", error
        with contextlib.suppress(Exception):  # Already failing; the first error is the one kept.
            stream.disconnect()


class _Joint:
    """A URDF joint's transform as a function of its position, vectorised over samples (Rerun's own
    ``compute_transform_columns`` took 12 ms per simulated second for Spot's 19 joints)."""

    def __init__(self, kind: str, parent: str, child: str, axis, origin_xyz, origin_rpy) -> None:
        self.kind, self.parent, self.child = kind, parent, child
        self.axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        self.origin = np.asarray(origin_xyz, dtype=float)
        self.origin_q = _quaternion_from_rpy(*origin_rpy)

    @classmethod
    def of(cls, joint) -> _Joint:
        parts = (joint.parent_link, joint.child_link, joint.axis, joint.origin_xyz, joint.origin_rpy)
        return cls(joint.joint_type, *parts)

    def transforms(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Translations [N, 3] and quaternions [N, 4] (x, y, z, w) of the child link in the parent's
        frame at joint positions ``values`` [N] (rad or m): the joint origin, then the motion."""
        values = np.asarray(values, dtype=float)
        count = len(values)
        if self.kind == "prismatic":
            local = self.axis[None, :] * values[:, None]
            translation = self.origin[None, :] + _rotate(self.origin_q, local)
            quaternion = np.tile(self.origin_q, (count, 1))
            return translation, quaternion
        half = values / 2.0
        motion = np.concatenate([self.axis[None, :] * np.sin(half)[:, None], np.cos(half)[:, None]], axis=1)
        return np.tile(self.origin, (count, 1)), _multiply(self.origin_q, motion)


def _path(*parts: str) -> str:
    """An entity path from its parts, each escaped (a body's USD path is one part)."""
    from rerun import escape_entity_path_part

    return "/" + "/".join(escape_entity_path_part(part) for part in parts)


def _moved(logged: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Bodies [B] whose pose moved past :data:`POSITION_STEP` or :data:`ROTATION_STEP` from ``logged``."""
    shift = np.linalg.norm(poses[:, :3] - logged[:, :3], axis=1)
    dot = np.abs(np.sum(poses[:, 3:] * logged[:, 3:], axis=1))
    turn = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    return (shift > POSITION_STEP) | (turn > ROTATION_STEP)


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw [rad] as a quaternion (x, y, z, w)."""
    cr, sr, cp, sp, cy, sy = (f(a / 2.0) for a in (roll, pitch, yaw) for f in (math.cos, math.sin))
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Quaternion products ``left * right`` (x, y, z, w); ``left`` [4], ``right`` [N, 4]."""
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right.T
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=1,
    )


def _rotate(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """``vectors`` [N, 3] rotated by ``quaternion`` (x, y, z, w)."""
    q, w = quaternion[:3], quaternion[3]
    t = 2.0 * np.cross(q[None, :], vectors)
    return vectors + w * t + np.cross(q[None, :], t)
