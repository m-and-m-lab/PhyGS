# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scenes by id: a spec names a scene (``scene: {id: grid_storage}``); this finds its USD file.

A scene id is a folder name. The folders are looked up, in order, under ``$PHYGS_SCENES``, under
``~/.cache/phygs/scenes`` (where the scene fetcher installs releases) and under the robot tests'
fixture scenes (``phygs_robot_tests/scenes``). A folder holds ``scene.usda``,
``scene.usdc`` or ``scene.usd``; a fetched release may instead be a record ``<id>.json`` whose
``usd_path`` points at the installed file. No spec ever holds a file path.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SCENES_ENV = "PHYGS_SCENES"
USER_CACHE = Path("~/.cache/phygs/scenes")
FIXTURES = Path(__file__).resolve().parents[2] / "phygs_robot_tests" / "scenes"  # The authored test scenes.
FILES = ("scene.usda", "scene.usdc", "scene.usd")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def valid_id(scene_id: str) -> bool:
    """A scene id is one folder name: no separators, never ``..``."""
    return isinstance(scene_id, str) and bool(ID.fullmatch(scene_id)) and ".." not in scene_id


def scene_roots() -> list[Path]:
    """Where scene folders are looked for, in order."""
    roots = []
    if os.environ.get(SCENES_ENV):
        roots.append(Path(os.environ[SCENES_ENV]).expanduser())
    roots += [USER_CACHE.expanduser(), FIXTURES]
    return roots


def scene_file(scene_id: str) -> Path:
    """The scene's USD file (absolute); ``FileNotFoundError`` names the roots searched."""
    if not valid_id(scene_id):
        raise ValueError(f"Invalid scene id {scene_id!r}: one folder name, no path separators or '..'.")
    for root in scene_roots():
        folder = root / scene_id
        for name in FILES:
            if (folder / name).is_file():
                return (folder / name).resolve()
        record = root / f"{scene_id}.json"
        if record.is_file():
            path = Path(json.loads(record.read_text())["usd_path"])
            if path.is_file():
                return path.resolve()
    searched = ", ".join(str(root) for root in scene_roots())
    raise FileNotFoundError(f"No scene {scene_id!r} under {searched}; set ${SCENES_ENV} or fetch the scene.")
