# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The live video's states (``--record``): the robot's samples and Nav2's routes the video's overview
strip draws while the run goes (:mod:`.trace`). In memory only; a run's states are in its recording
(:mod:`.log`, read back by :mod:`.reader`). This goes with the live video (D32)."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from phygs.recording.reader import States


class StateLog:
    """The samples (time, root pose, holding) and the routes so far, as :class:`~.reader.States`."""

    def __init__(self) -> None:
        self._time: list[float] = []
        self._root: list[np.ndarray] = []
        self._holding: list[bool] = []
        self._routes: list[tuple[float, tuple | None, np.ndarray]] = []
        self._route_key: tuple | None = None  # The last logged route, as bytes (NaN goals compare equal).
        self._route_given: tuple | None = None  # The last (goal, points) objects given.

    def __len__(self) -> int:
        return len(self._time)

    def sample(self, time_seconds: float, root: np.ndarray, *, holding: bool) -> None:
        """The robot now: its root link pose [x, y, z, qx, qy, qz, qw] (copied: the caller may hand a
        view of a buffer the simulation overwrites) and whether the gripper holds an object."""
        self._time.append(float(time_seconds))
        self._root.append(np.array(root, dtype=np.float32).reshape(7))
        self._holding.append(bool(holding))

    def route(self, time_seconds: float, goal: tuple | None, points: Sequence[Sequence[float]]) -> None:
        """Nav2's route when it differs from the last one (no goal and no points: none)."""
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

    def states(self) -> States:
        """The samples and routes so far (no joints, bodies, costmaps or plans)."""
        count = len(self._time)
        goals = np.full((len(self._routes), 3), np.nan, dtype=np.float32)
        for row, (_, goal, _) in enumerate(self._routes):
            if goal is not None:
                goals[row, : len(goal)] = goal
        return States(
            dict(
                time=np.asarray(self._time, dtype=np.float64),
                root=np.asarray(self._root, dtype=np.float32).reshape(-1, 7),
                joints=np.zeros((count, 0), np.float32),
                holding=np.asarray(self._holding, dtype=bool),
                command=np.full((count, 3), np.nan, np.float32),
                route_time=np.asarray([t for t, _, _ in self._routes], dtype=np.float64),
                route_goal=goals,
                route_start=np.cumsum([0] + [len(points) for _, _, points in self._routes]).astype(np.int64),
                route_points=np.concatenate([p for _, _, p in self._routes] or [np.zeros((0, 2), np.float32)]),
                joint_names=np.zeros(0, dtype=str),
                body_names=np.zeros(0, dtype=str),
                body_sample=np.zeros(0, np.int32),
                body_index=np.zeros(0, np.int32),
                body_pose=np.zeros((0, 7), np.float32),
                meta=np.asarray(json.dumps({})),
            )
        )
