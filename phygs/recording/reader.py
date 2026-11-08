# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An episode's recording (``episode.rrd``, :mod:`.log`) read back: its properties, the agent's audit
lines, the states ``phygs replay`` and :mod:`.trace` draw from (:class:`States`), the arm contacts'
forces over time, the arm planner's requests (:meth:`Recording.plan_requests`, which
``examples/replay_plans.py`` plans again), what the agent received and, at level ``all_frames``,
every camera capture (:meth:`Recording.images`, read lazily: images are not loaded with the rest).

Rows are put in order by their ``tick``, then by when they were logged (Rerun's row ids grow as rows
are logged), so rows of one tick keep their order. A recording whose run died while writing ends in a
partial message; it is read through Rerun's own repair, ``rerun rrd merge --continue-on-error``, which
keeps every whole message (:attr:`Recording.repaired`)."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phygs.recording import log as episode_log
from phygs.recording.panels import PlannerView

PLAN_ARRAYS = ("voxels", "arm_start", "arm_end", "end_spheres", "tool_path")  # PlannerView's array fields.

_SEPARATOR = re.compile(r"(?<!\\)/")  # Entity path parts are split at unescaped slashes.
_ESCAPED = re.compile(r"\\(.)")


class Recording:
    """``episode.rrd`` (the file, or the run folder that holds it), read once."""

    def __init__(self, path: Path) -> None:
        import pyarrow as pa
        import rerun as rr

        path = Path(path)
        self.path = path / episode_log.FILE if path.is_dir() else path
        self._rows: dict[tuple[str, ...], list[tuple[int, bytes, float, pa.RecordBatch, int]]] = {}
        self.properties: dict = {}
        self._folder = None  # The repaired copy's folder, kept while this object lives.
        try:
            chunks = rr.chunk.RrdReader(self.path).stream().drop(content=list(IMAGES)).to_chunks()
            self.repaired, self._source = False, self.path
        except RuntimeError:  # A partial message at the end: the run died while it wrote.
            self._folder = tempfile.TemporaryDirectory(prefix="phygs_recording_")
            self._source, self.repaired = _repaired(self.path, Path(self._folder.name)), True
            chunks = rr.chunk.RrdReader(self._source).stream().drop(content=list(IMAGES)).to_chunks()
        for chunk in chunks:
            entity = _parts(str(chunk.entity_path))
            batch = chunk.to_record_batch()
            if entity[:1] == (rr.RECORDING_PROPERTIES_PATH.strip("/"),):
                self._read_properties(entity, batch)
                continue
            if chunk.is_static:  # What is logged once to be drawn (the URDF, colliders, plot names).
                continue
            names = batch.schema.names
            ticks = batch.column("tick").to_pylist() if "tick" in names else [0] * batch.num_rows
            times = _seconds(batch.column("sim_time")) if "sim_time" in names else np.zeros(batch.num_rows)
            ids = batch.column("rerun.controls.RowId").to_pylist()
            rows = self._rows.setdefault(entity, [])
            rows.extend((ticks[row], ids[row], float(times[row]), batch, row) for row in range(batch.num_rows))
        for rows in self._rows.values():
            rows.sort(key=lambda row: (row[0], row[1]))
        schema = self.properties.get("phygs_log_schema")
        if schema != episode_log.SCHEMA:
            raise ValueError(f"{self.path}: recording schema {schema!r}; this PhyGS reads {episode_log.SCHEMA}.")

    @classmethod
    def load(cls, run: Path) -> Recording:
        return cls(run)

    def _read_properties(self, entity: tuple[str, ...], batch) -> None:
        for name in batch.schema.names:
            if name == "rerun.controls.RowId" or ":" in name:
                continue
            for cell in batch.column(name).to_pylist():
                if cell:  # A property sent as AnyValues(value=JSON) under /__properties/<name>.
                    key = entity[-1] if name == "value" and len(entity) > 1 else name
                    self.properties[key] = json.loads(cell[0])

    # --- Rows ---

    def rows(self, *entity: str) -> list[tuple[int, float, dict]]:
        """The rows of ``entity`` (its path parts, unescaped) in order: (tick, simulated time [s], {column: the
        row's values as a numpy array, or a list for nested lists and text})."""
        return [(tick, time, _row(batch, row)) for tick, _, time, batch, row in self._rows.get(tuple(entity), ())]

    def entities(self, *prefix: str) -> list[tuple[str, ...]]:
        """The entities under ``prefix`` that hold rows."""
        return [entity for entity in self._rows if entity[: len(prefix)] == prefix]

    @property
    def audit(self) -> list[dict]:
        """The agent's calls and steps (``phygs.session.Audit``'s lines), in the order they were logged."""
        rows = self._rows.get(("agent", "calls"), ())
        ordered = sorted(rows, key=lambda row: row[1])  # Row ids: the order of logging.
        return [json.loads(_row(batch, row)["line"][0]) for _, _, _, batch, row in ordered]

    # --- The states ---

    def states(self) -> States:
        """The states the run sampled, as :class:`States`, with the recording's ``replay``
        properties as its ``meta`` and the command spans."""
        layout, bodies = self.properties["robot_layout"], self.properties.get("scene_bodies", [])
        robot = layout["entity"]
        sampled = [("truth", robot, "base"), ("estimate", "holding"), ("control", "base_command")]
        counts = {entity: len(self._rows.get(entity, ())) for entity in sampled}
        count = min(counts.values())
        if not self.repaired and len(set(counts.values())) > 1:
            raise ValueError(f"{self.path}: the sampled series differ in length: {counts}")
        # A run that died may have sent one series' last batch and not another's: the samples all hold.
        base = self.rows("truth", robot, "base")[:count]
        ticks = np.array([tick for tick, _, _ in base], dtype=np.int64)

        def series(*entity: str, width: int = 1) -> np.ndarray:
            values = [row["Scalars:scalars"] for _, _, row in self.rows(*entity)[:count]]
            return np.array(values, dtype=np.float64).reshape(count, width)

        # The joints every physics tick: those of the samples' ticks (a run that died: the last ones known).
        ticked = self.rows("truth", robot, "joint_positions")
        tick_index = np.array([tick for tick, _, _ in ticked], dtype=np.int64)
        rows = np.clip(np.searchsorted(tick_index, ticks, side="right") - 1, 0, None)
        width = len(layout["joints"])
        joints = np.array([ticked[row][2]["Scalars:scalars"] for row in rows], dtype=np.float32).reshape(-1, width)
        arrays = dict(
            time=np.array([time for _, time, _ in base], dtype=np.float64),
            root=np.array(
                [np.concatenate([v["Transform3D:translation"][0], v["Transform3D:quaternion"][0]]) for _, _, v in base],
                dtype=np.float32,
            ).reshape(-1, 7),
            joints=joints,
            holding=series("estimate", "holding")[:, 0] > 0.5,
            command=series("control", "base_command", width=3).astype(np.float32),
            joint_names=np.asarray(layout["joints"], dtype=str),
            body_names=np.asarray(bodies, dtype=str),
        )
        arrays.update(self._bodies(bodies, ticks))
        arrays.update(self._routes())
        arrays.update(self._costmaps())
        arrays.update(self._plans())
        meta = dict(self.properties.get("replay", {}), commands=self.command_spans())
        arrays["meta"] = np.asarray(json.dumps(meta))
        return States(arrays)

    def contact_forces(self) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        """Each arm contact's force over time: (link, body) -> (times [s], forces [N]), at every
        contact read while it lasted and 0 when it ended."""
        found = {}
        for entity in self.entities("truth", "contact_forces"):
            rows = self.rows(*entity)
            found[(entity[2], entity[3])] = (
                np.array([time for _, time, _ in rows]),
                np.array([values["Scalars:scalars"][0] for _, _, values in rows]),
            )
        return found

    def command_spans(self) -> list[dict]:
        """The command timeline's spans in their order: each span's last logged state."""
        spans = {}
        for _, _, values in self.rows("control", "commands"):
            span = json.loads(values["span"][0])
            spans[span.pop("index")] = span
        return [spans[index] for index in sorted(spans)]

    def observations(self) -> list[tuple[float, dict]]:
        """What each observation the agent received held (simulated time [s], summary: stamp tick,
        each camera's capture tick, joints, base pose, IMU when permitted); its images: :meth:`images`."""
        return [(time, json.loads(values["packet"][0])) for _, time, values in self.rows("agent_view", "observe")]

    def images(self, branch: str, camera: str) -> Iterator[CameraImage]:
        """The images of ``camera`` in ``branch`` (``agent_view``: those the agent received;
        ``sensors``: every capture, level ``all_frames``) in capture order, read lazily."""
        import rerun as rr

        if (
            branch == "agent_view"
            and self.properties.get("settings", {}).get("recording", {}).get("level") == "all_frames"
        ):
            received = {summary["cameras"].get(camera) for _, summary in self.observations()}
            yield from (image for image in self.images("sensors", camera) if image.tick in received)
            return
        root = ("agent_view", "images") if branch == "agent_view" else (branch,)
        pending: dict[int, dict] = {}
        content = [f"/{'/'.join(root)}/{camera}/**", f"/{'/'.join(root)}/{camera}"]
        stream = rr.chunk.RrdReader(self._source).stream().filter(content=content, is_static=False)
        for chunk in stream:
            entity = _parts(str(chunk.entity_path))
            batch = chunk.to_record_batch()
            ticks = batch.column("tick").to_pylist()
            times = _seconds(batch.column("sim_time"))
            for row in range(batch.num_rows):
                values = _row(batch, row)
                entry = pending.setdefault(ticks[row], dict(tick=ticks[row], time=float(times[row])))
                kind = entity[-1] if len(entity) > len(root) + 1 else "pose"
                entry[kind] = values
        for tick in sorted(pending):  # The file holds an image's parts in any order; ticks put them together.
            entry = pending[tick]
            pose = entry.get("pose", {})
            yield CameraImage(
                tick=tick,
                time=entry["time"],
                rgb=_rgb(entry["rgb"]) if "rgb" in entry else None,
                depth=_depth(entry["depth"]) if "depth" in entry else None,
                intrinsics=np.asarray(pose["intrinsics"]).reshape(3, 3) if "intrinsics" in pose else None,
                body_from_camera=np.asarray(pose["body_from_camera"]).reshape(4, 4)
                if "body_from_camera" in pose
                else None,
            )

    def plan_requests(self) -> list[PlanRecord]:
        """Every arm plan's request, in the order they finished, with its depth images."""
        requests = self.rows("planner", "request")
        cameras = {entity[-1]: self.rows(*entity) for entity in self.entities("planner", "request", "depth")}
        found = []
        for index, (_, time, values) in enumerate(requests):
            request = json.loads(values["request"][0])
            names = [image["camera"] for image in request.get("images") or ()]
            depth = {name: _depth(cameras[name][index][2]) for name in names}
            found.append(PlanRecord(time, request, depth, json.loads(values["outcome"][0])))
        return found

    def _bodies(self, bodies: list[str], ticks: np.ndarray) -> dict[str, np.ndarray]:
        index = {name: position for position, name in enumerate(bodies)}
        rows = []  # (tick, row id, body index, pose)
        for entity in self.entities("truth", "scene", "bodies"):
            if len(entity) != 4:  # The body's colliders are below it.
                continue
            body = index[entity[3]]
            for tick, row_id, _, batch, row in self._rows[entity]:
                values = _row(batch, row)
                pose = np.concatenate([values["Transform3D:translation"][0], values["Transform3D:quaternion"][0]])
                rows.append((tick, row_id, body, pose))
        rows.sort(key=lambda row: (row[0], row[1]))
        # A body pose belongs to the sample logged in the same tick (the first, if a tick has two).
        samples = np.searchsorted(ticks, [tick for tick, *_ in rows], side="left").astype(np.int32)
        kept = samples < len(ticks)  # Poses after the last whole sample of a run that died.
        rows, samples = [row for row, keep in zip(rows, kept) if keep], samples[kept]
        return dict(
            body_sample=samples,
            body_index=np.array([body for _, _, body, _ in rows], dtype=np.int32),
            body_pose=np.array([pose for *_, pose in rows], dtype=np.float32).reshape(-1, 7),
        )

    def _routes(self) -> dict[str, np.ndarray]:
        rows = self.rows("nav", "route")
        points = [
            values["LineStrips3D:strips"][0][:, :2] if len(values["LineStrips3D:strips"]) else np.zeros((0, 2))
            for _, _, values in rows
        ]
        return dict(
            route_time=np.array([time for _, time, _ in rows], dtype=np.float64),
            route_goal=np.array([values["goal"] for _, _, values in rows], dtype=np.float32).reshape(-1, 3),
            route_start=np.cumsum([0] + [len(p) for p in points]).astype(np.int64),
            route_points=np.concatenate(points or [np.zeros((0, 2))]).astype(np.float32).reshape(-1, 2),
        )

    def _costmaps(self) -> dict[str, np.ndarray]:
        rows = self.rows("nav", "costmap")
        arrays = dict(
            costmap_time=np.array([time for _, time, _ in rows], dtype=np.float64),
            costmap_origin=np.array([values["origin"] for _, _, values in rows], dtype=np.float64).reshape(-1, 2),
            costmap_cell=np.array([values["cell"][0] for _, _, values in rows], dtype=np.float64),
        )
        for index, (_, _, values) in enumerate(rows):
            image = values["Image:format"][0]
            buffer = np.asarray(values["Image:buffer"][0], dtype=np.uint8)
            arrays[f"costmap{index}"] = buffer.reshape(image["height"], image["width"])
        return arrays

    def _plans(self) -> dict[str, np.ndarray]:
        plans = self.rows("planner", "plan")
        parts = {
            name: self.rows("planner", "plan", name)
            for name in ("voxels", "boxes", "arm_start", "arm_path", "arm_end", "end_spheres", "tool_path")
        }
        arrays = dict(
            plan_time=np.array([time for _, time, _ in plans], dtype=np.float64),
            plan_label=np.asarray([values["label"][0] for _, _, values in plans], dtype=str).reshape(-1),
            plan_voxel_size=np.array([values["voxel_size"][0] for _, _, values in plans], dtype=np.float64),
            plan_floor=np.array([values["floor"][0] for _, _, values in plans], dtype=np.float64),
        )
        for index in range(len(plans)):
            part = {name: rows[index][2] for name, rows in parts.items()}

            def strips(name: str) -> list[np.ndarray]:
                found = part[name]["LineStrips3D:strips"]
                return [np.asarray(strip, dtype=np.float32).reshape(-1, 3) for strip in found]

            def line(name: str) -> np.ndarray:
                found = strips(name)
                return found[0] if found else np.zeros((0, 3), np.float32)

            spheres, boxes = part["end_spheres"], part["boxes"]
            radii = np.asarray(spheres.get("Points3D:radii", np.zeros(0)), np.float32).reshape(-1, 1)
            centers = _points(spheres.get("Points3D:positions"))
            frames = strips("arm_path")
            arrays.update(
                {
                    f"plan{index}_voxels": _points(part["voxels"].get("Points3D:positions")),
                    f"plan{index}_boxes": np.concatenate(
                        [_points(boxes.get("Boxes3D:centers")), 2 * _points(boxes.get("Boxes3D:half_sizes"))], axis=1
                    ),
                    f"plan{index}_arm_start": line("arm_start"),
                    f"plan{index}_arm_end": line("arm_end"),
                    f"plan{index}_tool_path": line("tool_path"),
                    f"plan{index}_end_spheres": np.concatenate([centers, radii], axis=1),
                    f"plan{index}_arm_path": np.concatenate(frames) if frames else np.zeros((0, 3), np.float32),
                    f"plan{index}_arm_path_sizes": np.array([len(frame) for frame in frames], dtype=np.int64),
                }
            )
        return arrays


IMAGES = ("/agent_view/images/**", "/sensors/**")  # Image entities: read only through Recording.images.


@dataclass(frozen=True)
class CameraImage:
    """One camera image from a recording: capture tick and time [s], RGB [H, W, 3] uint8 and depth
    [H, W] [m] (0: no depth) as the camera gave them, and (``sensors`` only) its intrinsics [3, 3]
    and pose in the body frame [4, 4]."""

    tick: int
    time: float
    rgb: np.ndarray | None
    depth: np.ndarray | None
    intrinsics: np.ndarray | None = None
    body_from_camera: np.ndarray | None = None


@dataclass(frozen=True)
class PlanRecord:
    """One arm plan as the planner worker got it: when it finished [s sim], the request (JSON, as
    ``PlanRequest.to_json`` gives it, each depth image named by its ``camera`` instead of a file),
    the depth images [m, float32] by camera and the outcome (success, reason, plan time)."""

    time: float
    request: dict
    depth: dict[str, np.ndarray]
    outcome: dict


def _repaired(path: Path, folder: Path) -> Path:
    """A copy in ``folder`` of a recording that ends in a partial message (a run killed while writing),
    with every whole message, written again by Rerun's command line."""
    repaired = folder / path.name
    command = [sys.executable, "-m", "rerun", "rrd", "merge", "--continue-on-error", str(path), "-o", str(repaired)]
    subprocess.run(command, check=True, capture_output=True, timeout=600)
    return repaired


def _parts(entity_path: str) -> tuple[str, ...]:
    """An entity path's parts, unescaped (``/truth/scene/bodies/\\/World\\/box`` has 4, the last
    ``/World/box``)."""
    return tuple(_ESCAPED.sub(r"\1", part) for part in _SEPARATOR.split(entity_path) if part)


def _seconds(column) -> np.ndarray:
    import pyarrow as pa

    return column.cast(pa.int64()).to_numpy(zero_copy_only=False) / 1e9


def _row(batch, row: int) -> dict:
    """One row's columns: component batches as numpy arrays (lists of arrays for nested lists, Python
    values for text and structs)."""
    import pyarrow as pa

    values = {}
    for name, column in zip(batch.schema.names, batch.columns):
        if name in ("rerun.controls.RowId", "tick", "sim_time", "wall_time", "seq", "log_time", "log_tick"):
            continue
        cell = column[row]
        if not cell.is_valid:
            continue
        array = cell.values
        kind = array.type
        if pa.types.is_fixed_size_list(kind):
            values[name] = array.flatten().to_numpy(zero_copy_only=False).reshape(len(array), kind.list_size)
        elif pa.types.is_list(kind) and pa.types.is_fixed_size_list(kind.value_type):
            size = kind.value_type.list_size
            values[name] = [
                (item.values.flatten().to_numpy(zero_copy_only=False).reshape(-1, size) if item.is_valid else None)
                for item in array
            ]
        elif pa.types.is_floating(kind) or pa.types.is_integer(kind):
            values[name] = array.to_numpy(zero_copy_only=False)
        elif pa.types.is_list(kind) and (pa.types.is_integer(kind.value_type) or pa.types.is_floating(kind.value_type)):
            values[name] = [item.values.to_numpy(zero_copy_only=False) if item.is_valid else None for item in array]
        else:
            values[name] = array.to_pylist()
    return values


def _rgb(values: dict) -> np.ndarray:
    """A logged RGB image as the uint8 array [H, W, 3] it was."""
    image = values["Image:format"][0]
    return np.asarray(values["Image:buffer"][0], np.uint8).reshape(image["height"], image["width"], 3)


def _depth(values: dict) -> np.ndarray:
    """A logged depth image [m] as the float32 array it was."""
    image = values["DepthImage:format"][0]
    return np.frombuffer(np.asarray(values["DepthImage:buffer"][0], np.uint8).tobytes(), np.float32).reshape(
        image["height"], image["width"]
    )


def _points(values) -> np.ndarray:
    return np.zeros((0, 3), np.float32) if values is None else np.asarray(values, dtype=np.float32).reshape(-1, 3)


@dataclass(frozen=True)
class Route:
    """Nav2's route from ``time`` [s] on: the goal (x, y [m], heading [rad] or None) and the points
    [P, 2] [m]; no goal and no points when no navigation command runs."""

    time: float
    goal: tuple[float, float, float | None] | None
    points: np.ndarray


class States:
    """A run's states (:meth:`Recording.states`, or the live video's, :mod:`.state_log`): the samples
    every camera period, the scene at any sample, Nav2's routes and costmaps and the arm plans at any
    time, the command spans and ``meta`` (what replay needs to build the scene again)."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.meta = json.loads(str(arrays["meta"]))
        self.time = arrays["time"]
        self.root, self.joints, self.holding = arrays["root"], arrays["joints"], arrays["holding"]
        self.command = arrays["command"]
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
            raise ValueError("The states' first sample does not hold every body.")
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
