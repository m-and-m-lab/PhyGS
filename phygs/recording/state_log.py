# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A run's state log (``state.npz``): what moved while the episode ran, enough to render it again.

The World samples the state every camera period [s sim] while the agent runs, rendering nothing:
the simulated time, the robot's root link pose and joint positions, whether the gripper holds
something, and the pose of every scene body that moved since its last logged pose (more than
:data:`POSITION_STEP` or :data:`ROTATION_STEP`). The first sample logs every body, so the scene at
a sample is each body's last logged pose. Each sample also holds the base's commanded velocity.
Nav2's route and global costmap are logged when they change, every new arm plan's display (the
recording's arm planner panel, :class:`~phygs.recording.panels.PlannerView`) when it finishes, and
the command timeline's spans when the log is written. ``phygs replay`` renders the run from this
file, with the recording's panels (:mod:`phygs.world.replay`); :mod:`.trace` reads the timeline and
the robot's path from it.

Arrays of ``state.npz`` (S samples, J joints, B bodies, R logged body poses, K routes, P route points,
C costmaps, N plans):

==================  ============  ===================================================================
``time``            [S]           simulated time [s]
``root``            [S, 7]        robot root link pose [m, quaternion xyzw]
``joints``          [S, J]        joint positions [m or rad, depending on joint type]
``holding``         [S]           whether the gripper holds an object
``command``         [S, 3]        the base's commanded body velocity [m/s, m/s, rad/s] (NaN: none)
``body_sample``     [R]           the sample each logged body pose belongs to
``body_index``      [R]           the body, an index into ``body_names``
``body_pose``       [R, 7]        the body's world link pose [m, quaternion xyzw]
``route_time``      [K]           when the route changed [s]
``route_goal``      [K, 3]        the goal x, y [m] and heading [rad] (NaN: none); all NaN: no route
``route_start``     [K + 1]       each route's first point in ``route_points``
``route_points``    [P, 2]        route points x, y [m]
``costmap_time``    [C]           when Nav2's global costmap changed [s]
``costmap_origin``  [C, 2]        its origin x, y [m]
``costmap_cell``    [C]           its cell size [m]
``costmap<c>``      [rows, cols]  costmap ``c`` (Nav2 costs 0-255; row 0 at the lowest y)
``plan_time``       [N]           when an arm plan finished [s]
``plan_label``      [N]           its one-line label
``plan_voxel_size`` [N]           its obstacle voxels' size [m]
``plan_floor``      [N]           the floor height in the body frame [m] (NaN: unknown)
``plan<n>_<field>`` [., 3 or 4]   plan ``n``'s ``voxels``, ``boxes`` (centre, size), ``arm_start``,
                                  ``arm_path`` (its frames joined; ``arm_path_sizes``: their lengths),
                                  ``arm_end``, ``end_spheres`` and ``tool_path``, body frame [m]
``joint_names``     [J]           the robot's joint names
``body_names``      [B]           the scene bodies' USD paths
``meta``            []            JSON: :data:`SCHEMA`, the sample period, the scene, robot and light
                                  the replay needs, and the command spans (``commands``)
==================  ============  ===================================================================

Logs written before PhyGS 0.9.0 have no ``command``, costmaps or plans; they read back without them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phygs.recording.panels import PlannerView

SCHEMA = 1
POSITION_STEP = 0.001  # [m]; a body moved this far since its last logged pose is logged again.
ROTATION_STEP = math.radians(0.25)  # [rad]; likewise for its rotation.
FILE = "state.npz"
COSTMAP_PERIOD = 1.0  # [s sim]; Nav2's costmap is read at most this often (as the recording's panel).
PLAN_ARRAYS = ("voxels", "arm_start", "arm_end", "end_spheres", "tool_path")  # PlannerView's array fields.


@dataclass(frozen=True)
class Route:
    """Nav2's route from ``time`` [s] on: the goal (x, y [m], heading [rad] or None) and the points
    [P, 2] [m]; no goal and no points when no navigation command runs."""

    time: float
    goal: tuple[float, float, float | None] | None
    points: np.ndarray


class StateLog:
    """Collects samples in memory while the episode runs and writes ``state.npz`` once (:meth:`write`).

    ``joint_names`` and ``body_names`` fix the order of every sample's ``joints`` and ``bodies``;
    ``meta`` is stored as given (JSON-friendly) beside the arrays.
    """

    def __init__(self, path: Path, *, joint_names: Sequence[str], body_names: Sequence[str], meta: dict) -> None:
        self.path = Path(path)
        self.joint_names, self.body_names = tuple(joint_names), tuple(body_names)
        self.meta = dict(meta)
        self._time: list[float] = []
        self._root: list[np.ndarray] = []
        self._joints: list[np.ndarray] = []
        self._holding: list[bool] = []
        self._command: list[np.ndarray] = []
        self._body_rows: list[tuple[int, np.ndarray, np.ndarray]] = []  # (sample, indices, poses)
        self._logged: np.ndarray | None = None  # Each body's last logged pose [B, 7].
        self._routes: list[tuple[float, tuple, np.ndarray]] = []
        self._route_key: tuple | None = None  # The last logged route, as bytes (NaN goals compare equal).
        self._route_given: tuple | None = None  # The last (goal, points) objects given.
        self._costmaps: list[tuple[float, np.ndarray, tuple[float, float], float]] = []
        self._costmap_read = -math.inf  # Simulated time [s] of the last costmap read (wants_costmap).
        self._plans: list[tuple[float, PlannerView]] = []
        self.written = False

    def __len__(self) -> int:
        return len(self._time)

    def sample(
        self,
        time_seconds: float,
        root: np.ndarray,
        joints: np.ndarray,
        bodies: np.ndarray,
        *,
        holding: bool,
        command: Sequence[float] | None = None,
    ) -> None:
        """Log the state now: the root pose [7], joint positions [J], every body's pose [B, 7] and the
        base's commanded velocity [m/s, m/s, rad/s] (copied: the caller may hand views of buffers the
        simulation overwrites)."""
        root = np.array(root, dtype=np.float32).reshape(7)
        joints = np.array(joints, dtype=np.float32).reshape(len(self.joint_names))
        bodies = np.asarray(bodies, dtype=np.float32).reshape(len(self.body_names), 7)
        if self._logged is None:
            moved = np.arange(len(bodies))
        else:
            moved = np.flatnonzero(_moved(self._logged, bodies))
        if len(moved):
            if self._logged is None:
                self._logged = bodies.copy()
            else:
                self._logged[moved] = bodies[moved]
            self._body_rows.append((len(self._time), moved.astype(np.int32), bodies[moved].copy()))
        self._time.append(float(time_seconds))
        self._root.append(root)
        self._joints.append(joints)
        self._holding.append(bool(holding))
        self._command.append(np.full(3, np.nan, np.float32) if command is None else np.array(command, np.float32))

    def route(self, time_seconds: float, goal: tuple | None, points: Sequence[Sequence[float]]) -> None:
        """Log Nav2's route when it differs from the last one logged (no goal and no points: none)."""
        if self._route_given is not None and goal == self._route_given[0] and points is self._route_given[1]:
            return  # The same route object as last time (Nav2's backend keeps it until it replans).
        self._route_given = (goal, points)
        goal = None if goal is None else tuple(float("nan") if value is None else float(value) for value in goal)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        key = (np.asarray(goal if goal is not None else (), dtype=np.float32).tobytes(), points.tobytes())
        if key == self._route_key:
            return
        self._route_key = key
        self._routes.append((float(time_seconds), goal, points))

    def wants_costmap(self, time_seconds: float) -> bool:
        """Whether to read Nav2's costmap now: :data:`COSTMAP_PERIOD` after the last read (or time
        went back: a new episode)."""
        if time_seconds - self._costmap_read >= COSTMAP_PERIOD or time_seconds < self._costmap_read:
            self._costmap_read = time_seconds
            return True
        return False

    def costmap(self, time_seconds: float, costs: np.ndarray, origin: Sequence[float], cell: float) -> None:
        """Log Nav2's global costmap (costs 0-255 [rows, cols], origin [m], cell size [m]) when it
        differs from the last one logged."""
        costs = np.asarray(costs, dtype=np.uint8)
        origin = (float(origin[0]), float(origin[1]))
        if self._costmaps:
            _, last, last_origin, last_cell = self._costmaps[-1]
            if last_origin == origin and last_cell == float(cell) and np.array_equal(last, costs):
                return
        self._costmaps.append((float(time_seconds), costs.copy(), origin, float(cell)))

    def plan(self, time_seconds: float, view: PlannerView) -> None:
        """Log an arm plan's display when it is a new one (the planner makes one per finished plan)."""
        if self._plans and self._plans[-1][1] is view:
            return
        self._plans.append((float(time_seconds), view))

    def write(self, commands: Sequence[dict] = ()) -> Path:
        """Write ``state.npz`` with the command ``commands`` (lane, name, start, end, status) in ``meta``."""
        rows = self._body_rows
        count = sum(len(indices) for _, indices, _ in rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            **self._costmap_arrays(),
            **self._plan_arrays(),
            **self._sample_arrays(commands),
            body_sample=np.concatenate([np.full(len(i), s, np.int32) for s, i, _ in rows] or [np.zeros(0, np.int32)]),
            body_index=np.concatenate([i for _, i, _ in rows] or [np.zeros(0, np.int32)]),
            body_pose=np.concatenate([p for _, _, p in rows] or [np.zeros((0, 7), np.float32)]).reshape(count, 7),
            body_names=np.asarray(self.body_names, dtype=str),
        )
        self.written = True
        return self.path

    def states(self) -> States:
        """The samples and routes so far, read back without the scene's bodies, costmaps and plans:
        what a recording's trace needs while the run goes (:mod:`.trace`)."""
        return States(
            dict(
                self._sample_arrays(()),
                body_sample=np.zeros(0, np.int32),
                body_index=np.zeros(0, np.int32),
                body_pose=np.zeros((0, 7), np.float32),
                body_names=np.zeros(0, dtype=str),
            )
        )

    def _sample_arrays(self, commands: Sequence[dict]) -> dict[str, np.ndarray]:
        """The samples, the routes, the joint names and ``meta`` (with ``commands``)."""
        goals = np.full((len(self._routes), 3), np.nan, dtype=np.float32)
        for row, (_, goal, _) in enumerate(self._routes):
            if goal is not None:
                goals[row, : len(goal)] = goal
        starts = np.cumsum([0] + [len(points) for _, _, points in self._routes]).astype(np.int64)
        return dict(
            time=np.asarray(self._time, dtype=np.float64),
            root=np.asarray(self._root, dtype=np.float32).reshape(-1, 7),
            joints=np.asarray(self._joints, dtype=np.float32).reshape(-1, len(self.joint_names)),
            holding=np.asarray(self._holding, dtype=bool),
            command=np.asarray(self._command, dtype=np.float32).reshape(-1, 3),
            route_time=np.asarray([t for t, _, _ in self._routes], dtype=np.float64),
            route_goal=goals,
            route_start=starts,
            route_points=np.concatenate([p for _, _, p in self._routes] or [np.zeros((0, 2), np.float32)]),
            joint_names=np.asarray(self.joint_names, dtype=str),
            meta=np.asarray(json.dumps(dict(self.meta, schema=SCHEMA, commands=list(commands)))),
        )

    def _costmap_arrays(self) -> dict[str, np.ndarray]:
        arrays = dict(
            costmap_time=np.asarray([time for time, *_ in self._costmaps], dtype=np.float64),
            costmap_origin=np.asarray([origin for _, _, origin, _ in self._costmaps], dtype=np.float64).reshape(-1, 2),
            costmap_cell=np.asarray([cell for *_, cell in self._costmaps], dtype=np.float64),
        )
        arrays.update({f"costmap{index}": costs for index, (_, costs, _, _) in enumerate(self._costmaps)})
        return arrays

    def _plan_arrays(self) -> dict[str, np.ndarray]:
        arrays = dict(
            plan_time=np.asarray([time for time, _ in self._plans], dtype=np.float64),
            plan_label=np.asarray([view.label for _, view in self._plans], dtype=str).reshape(-1),
            plan_voxel_size=np.asarray([view.voxel_size for _, view in self._plans], dtype=np.float64),
            plan_floor=np.asarray([np.nan if view.floor is None else view.floor for _, view in self._plans]),
        )
        for index, (_, view) in enumerate(self._plans):
            for field in PLAN_ARRAYS:
                arrays[f"plan{index}_{field}"] = np.asarray(getattr(view, field), dtype=np.float32)
            arrays[f"plan{index}_boxes"] = np.asarray(
                [(*center, *size) for center, size in view.boxes], dtype=np.float32
            ).reshape(-1, 6)
            frames = [np.asarray(frame, dtype=np.float32).reshape(-1, 3) for frame in view.arm_path]
            arrays[f"plan{index}_arm_path"] = np.concatenate(frames) if frames else np.zeros((0, 3), np.float32)
            arrays[f"plan{index}_arm_path_sizes"] = np.asarray([len(frame) for frame in frames], dtype=np.int64)
        return arrays


def _moved(logged: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Bodies [B] whose pose moved past :data:`POSITION_STEP` or :data:`ROTATION_STEP` from ``logged``."""
    shift = np.linalg.norm(poses[:, :3] - logged[:, :3], axis=1)
    dot = np.abs(np.sum(poses[:, 3:] * logged[:, 3:], axis=1))
    turn = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    return (shift > POSITION_STEP) | (turn > ROTATION_STEP)


class States:
    """A written state log, read back (:meth:`load`): the samples, the scene at any sample, the routes."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        meta = json.loads(str(arrays["meta"]))
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"State log schema {meta.get('schema')!r}; this PhyGS reads schema {SCHEMA}.")
        self.meta = meta
        self.time = arrays["time"]
        self.root, self.joints, self.holding = arrays["root"], arrays["joints"], arrays["holding"]
        self.command = arrays.get("command", np.full((len(self.time), 3), np.nan, dtype=np.float32))
        self._arrays = arrays
        self.costmap_times = arrays.get("costmap_time", np.zeros(0))
        self.plan_times = arrays.get("plan_time", np.zeros(0))
        self._plan_views: dict[int, PlannerView] = {}
        self.joint_names = tuple(str(name) for name in arrays["joint_names"])
        self.body_names = tuple(str(name) for name in arrays["body_names"])
        self._body_sample, self._body_index = arrays["body_sample"], arrays["body_index"]
        self._body_pose = arrays["body_pose"]
        starts, points = arrays["route_start"], arrays["route_points"]
        self.routes = tuple(
            Route(
                float(time),
                None if np.all(np.isnan(goal)) else (float(goal[0]), float(goal[1]), _number(goal[2])),
                points[starts[row] : starts[row + 1]],
            )
            for row, (time, goal) in enumerate(zip(arrays["route_time"], arrays["route_goal"]))
        )

    @classmethod
    def load(cls, path: Path) -> States:
        """Read ``state.npz`` (a file, or the run folder that holds it)."""
        path = Path(path)
        if path.is_dir():
            path = path / FILE
        with np.load(path, allow_pickle=False) as arrays:
            return cls({key: arrays[key] for key in arrays.files})

    def __len__(self) -> int:
        return len(self.time)

    @property
    def commands(self) -> list[dict]:
        """The command timeline's spans: lane, name, start and end [s] (None: still running), status."""
        return list(self.meta.get("commands", ()))

    def index(self, time_seconds: float) -> int:
        """The last sample at or before ``time_seconds`` (the first sample before it starts)."""
        return max(0, int(np.searchsorted(self.time, time_seconds, side="right")) - 1)

    def bodies_at(self, sample: int) -> np.ndarray:
        """Every body's pose [B, 7] at ``sample``: its last pose logged at or before it."""
        if not 0 <= sample < len(self):
            raise IndexError(f"Sample {sample} of {len(self)}.")
        rows = np.flatnonzero(self._body_sample <= sample)
        last = np.full(len(self.body_names), -1, dtype=np.int64)
        np.maximum.at(last, self._body_index[rows], rows)
        if np.any(last < 0):
            raise ValueError("The state log's first sample does not hold every body.")
        return self._body_pose[last].copy()

    def bodies(self) -> Iterator[np.ndarray]:
        """Every body's pose [B, 7] at each sample in order (a view updated in place: copy to keep)."""
        poses = np.full((len(self.body_names), 7), np.nan, dtype=np.float32)
        row, rows = 0, len(self._body_sample)
        for sample in range(len(self)):
            end = row
            while end < rows and self._body_sample[end] == sample:
                end += 1
            poses[self._body_index[row:end]] = self._body_pose[row:end]
            row = end
            yield poses

    def route_at(self, time_seconds: float) -> Route | None:
        """The route logged last at or before ``time_seconds``; None before any."""
        found = None
        for route in self.routes:
            if route.time > time_seconds:
                break
            found = route
        return found

    def costmap_at(self, time_seconds: float) -> tuple[np.ndarray, tuple[float, float], float] | None:
        """Nav2's global costmap logged last at or before ``time_seconds`` (costs, origin [m], cell
        size [m]); None before any."""
        index = int(np.searchsorted(self.costmap_times, time_seconds, side="right")) - 1
        if index < 0:
            return None
        origin = self._arrays["costmap_origin"][index]
        return (
            self._arrays[f"costmap{index}"],
            (float(origin[0]), float(origin[1])),
            float(self._arrays["costmap_cell"][index]),
        )

    def plan_at(self, time_seconds: float) -> PlannerView | None:
        """The arm plan's display logged last at or before ``time_seconds``; None before any."""
        index = int(np.searchsorted(self.plan_times, time_seconds, side="right")) - 1
        if index < 0:
            return None
        if index not in self._plan_views:
            arrays, key = self._arrays, f"plan{index}_"
            path, sizes = arrays[key + "arm_path"], arrays[key + "arm_path_sizes"]
            ends = np.cumsum(sizes)
            self._plan_views[index] = PlannerView(
                label=str(arrays["plan_label"][index]),
                voxel_size=float(arrays["plan_voxel_size"][index]),
                floor=_number(float(arrays["plan_floor"][index])),
                boxes=tuple((tuple(box[:3].tolist()), tuple(box[3:].tolist())) for box in arrays[key + "boxes"]),
                arm_path=tuple(path[end - size : end] for size, end in zip(sizes, ends)),
                **{field: arrays[key + field] for field in PLAN_ARRAYS},
            )
        return self._plan_views[index]

    def base_path(self, rate: float = 10.0) -> list[tuple[float, float, float, float]]:
        """The robot's (time [s], x, y [m], yaw [rad]) at ``rate`` [1/s]."""
        if not len(self):
            return []
        times = np.arange(self.time[0], self.time[-1] + 1e-9, 1.0 / rate)
        rows = self.root[np.maximum(0, np.searchsorted(self.time, times, side="right") - 1)].astype(float)
        qx, qy, qz, qw = rows[:, 3], rows[:, 4], rows[:, 5], rows[:, 6]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        path = np.round(np.column_stack([times, rows[:, 0], rows[:, 1], yaw]), 3)
        return [tuple(row) for row in path.tolist()]


def _number(value: float) -> float | None:
    return None if math.isnan(value) else float(value)
