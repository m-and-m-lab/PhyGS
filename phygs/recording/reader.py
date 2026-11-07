# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An episode's recording (``episode.rrd``, :mod:`.log`) read back: its properties, the agent's audit
lines and the states ``phygs replay`` and :mod:`.trace` draw from (:class:`~.state_log.States`).

Rows are put in order by their ``tick``, then by when they were logged (Rerun's row ids grow as rows
are logged), so rows of one tick keep their order. A recording whose run died while writing ends in a
partial message; it is read through Rerun's own repair, ``rerun rrd merge --continue-on-error``, which
keeps every whole message (:attr:`Recording.repaired`)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from phygs.recording import log as episode_log
from phygs.recording import state_log
from phygs.recording.state_log import States

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
        try:
            chunks = list(rr.chunk.RrdReader(self.path).stream().to_chunks())
            self.repaired = False
        except RuntimeError:  # A partial message at the end: the run died while it wrote.
            chunks, self.repaired = _repaired(self.path), True
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
        """The agent's calls and steps, as ``audit.jsonl`` held them, in the order they were logged."""
        rows = self._rows.get(("agent", "calls"), ())
        ordered = sorted(rows, key=lambda row: row[1])  # Row ids: the order of logging.
        return [json.loads(_row(batch, row)["line"][0]) for _, _, _, batch, row in ordered]

    # --- The states ---

    def states(self) -> States:
        """The states the run sampled, as :class:`~.state_log.States`, with the recording's ``replay``
        properties as its ``meta`` and the command spans."""
        layout, bodies = self.properties["robot_layout"], self.properties.get("scene_bodies", [])
        robot = layout["entity"]
        sampled = [("truth", robot, "base"), ("estimate", "holding"), ("control", "base_command")]
        sampled += [("truth", robot, "joint_positions")]
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

        joints = series("truth", robot, "joint_positions", width=len(layout["joints"])).astype(np.float32)
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
        spans = self.rows("control", "commands")
        commands = json.loads(spans[-1][2]["spans"][0]) if spans else []
        meta = dict(self.properties.get("replay", {}), schema=state_log.SCHEMA, commands=commands)
        arrays["meta"] = np.asarray(json.dumps(meta))
        return States(arrays)

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


def _repaired(path: Path) -> list:
    """The chunks of the whole messages of a recording that ends in a partial one (a run killed while
    writing), through Rerun's command line, which writes them again."""
    import rerun as rr

    with tempfile.TemporaryDirectory() as folder:
        repaired = Path(folder) / path.name
        command = [sys.executable, "-m", "rerun", "rrd", "merge", "--continue-on-error", str(path), "-o", str(repaired)]
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        return list(rr.chunk.RrdReader(repaired).stream().to_chunks())


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


def _points(values) -> np.ndarray:
    return np.zeros((0, 3), np.float32) if values is None else np.asarray(values, dtype=np.float32).reshape(-1, 3)
