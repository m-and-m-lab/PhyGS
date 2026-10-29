# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A scene's floor plan: one image per floor, seen straight down, and how world x, y map to its pixels.

A scene release may ship its plan beside its scene file, ``plan/plan.json`` with the images it
names; else ``phygs replay`` renders one and keeps it in the plan cache (``~/.cache/phygs/plans``,
or ``$PHYGS_PLANS``), under the scene's SHA-256 (the report's ``provenance.scene_usd.sha256``).
``plan.json`` (:data:`SCHEMA`)::

    {"schema": 1, "floors": [{"name": "ground", "image": "ground.png", "elevation": 0.0, "cut": 2.4,
                              "width": 1800, "height": 1240, "center": [11.2, 5.6], "ppm": 64.42}]}

A floor's image is the scene cut open above ``cut`` [m] (everything higher is left out) and the
mapping holds on its floor plane, at ``elevation`` [m]: world (x, y) [m] is pixel
(``width`` / 2 + (x - cx) ``ppm``, ``height`` / 2 - (y - cy) ``ppm``), +x right and +y up, ``ppm`` in
pixels per metre. No Isaac Sim imports.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCHEMA = 1
FILE = "plan.json"
RELEASE_FOLDER = "plan"  # Beside a release's scene file.


def cache_root() -> Path:
    """Where rendered plans are kept: ``$PHYGS_PLANS``, else ``~/.cache/phygs/plans``."""
    return Path(os.environ.get("PHYGS_PLANS", Path.home() / ".cache" / "phygs" / "plans"))


@dataclass(frozen=True)
class Floor:
    """One floor's plan image and its pixel mapping (see the module docstring)."""

    name: str
    image: str
    elevation: float
    cut: float
    width: int
    height: int
    center: tuple[float, float]
    ppm: float

    def pixel(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Pixel column and row of world x, y [m] (scalars or arrays) on this floor's image."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return self.width / 2 + (x - self.center[0]) * self.ppm, self.height / 2 - (y - self.center[1]) * self.ppm

    def world(self, column, row) -> tuple[np.ndarray, np.ndarray]:
        """World x, y [m] on the floor plane of pixel ``column``, ``row``."""
        column, row = np.asarray(column, dtype=float), np.asarray(row, dtype=float)
        return self.center[0] + (column - self.width / 2) / self.ppm, self.center[1] - (
            row - self.height / 2
        ) / self.ppm


@dataclass(frozen=True)
class Plan:
    """A scene's floors (:class:`Floor`), lowest first, with the folder their images are in."""

    folder: Path
    floors: tuple[Floor, ...]

    @classmethod
    def load(cls, path: Path) -> Plan:
        """Read ``plan.json`` (a file, or the folder that holds it) and check its images exist."""
        path = Path(path)
        if path.is_dir():
            path = path / FILE
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path}: plan schema {data.get('schema')!r}; this PhyGS reads schema {SCHEMA}.")
        floors = []
        for entry in data["floors"]:
            floor = Floor(
                name=str(entry["name"]),
                image=str(entry["image"]),
                elevation=float(entry["elevation"]),
                cut=float(entry["cut"]),
                width=int(entry["width"]),
                height=int(entry["height"]),
                center=(float(entry["center"][0]), float(entry["center"][1])),
                ppm=float(entry["ppm"]),
            )
            if Path(floor.image).is_absolute() or ".." in Path(floor.image).parts:
                raise ValueError(f"{path}: floor {floor.name!r} names an image outside its folder: {floor.image}")
            if not (path.parent / floor.image).is_file():
                raise FileNotFoundError(f"{path}: floor {floor.name!r} has no image {floor.image}")
            if floor.ppm <= 0 or floor.width <= 0 or floor.height <= 0:
                raise ValueError(f"{path}: floor {floor.name!r} needs a positive size and ppm")
            floors.append(floor)
        if not floors:
            raise ValueError(f"{path}: a plan needs at least one floor")
        return cls(path.parent, tuple(sorted(floors, key=lambda floor: floor.elevation)))

    def save(self) -> Path:
        """Write ``plan.json`` into :attr:`folder` (the images must be there already)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        floors = [dict(asdict(floor), center=list(floor.center)) for floor in self.floors]
        path = self.folder / FILE
        path.write_text(json.dumps(dict(schema=SCHEMA, floors=floors), indent=1) + "\n")
        return path

    def floor_at(self, z: float) -> Floor:
        """The floor a point at height ``z`` [m] stands on: the highest at or below it (the lowest
        floor for a point below every floor)."""
        below = [floor for floor in self.floors if floor.elevation <= z]
        return below[-1] if below else self.floors[0]

    def image_path(self, floor: Floor) -> Path:
        return self.folder / floor.image


def find(scene_usd: Path | str, sha256: str | None = None, *, cache: Path | None = None) -> Plan | None:
    """The scene's plan: the release's (``plan/`` beside the scene file), else the cached one for
    ``sha256``; None when neither exists."""
    release = Path(scene_usd).parent / RELEASE_FOLDER / FILE
    if release.is_file():
        return Plan.load(release)
    if sha256:
        cached = cached_folder(sha256, cache=cache) / FILE
        if cached.is_file():
            return Plan.load(cached)
    return None


def cached_folder(sha256: str, *, cache: Path | None = None) -> Path:
    """The cache folder of the scene with SHA-256 ``sha256``."""
    if not sha256 or not all(c in "0123456789abcdef" for c in sha256):
        raise ValueError(f"Not a SHA-256 hex digest: {sha256!r}")
    return (cache or cache_root()) / sha256


def fit(bounds: tuple[float, float, float, float], width: int, height: int, margin: float) -> tuple[tuple, float]:
    """The centre (x, y) [m] and ``ppm`` that fit ``bounds`` (xmin, ymin, xmax, ymax) [m], grown by
    ``margin`` [m], into a ``width`` x ``height`` image."""
    xmin, ymin, xmax, ymax = bounds
    span_x, span_y = xmax - xmin + 2 * margin, ymax - ymin + 2 * margin
    ppm = min(width / span_x, height / span_y)
    return ((xmin + xmax) / 2, (ymin + ymax) / 2), ppm
