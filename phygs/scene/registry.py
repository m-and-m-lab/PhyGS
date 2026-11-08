# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scenes by id: a spec names a scene (``scene: {id: grid_storage}``); this finds its USD file.

A scene id is a folder name. The folders are looked up, in order, under ``$PHYGS_SCENES``, under
``~/.cache/phygs/scenes`` (where the scene fetcher installs releases) and under the built-in scenes
that ship with PhyGS (``phygs/scene/builtin``: grid storage and the two navigation courses, the
robot tests' fixtures). A folder holds ``scene.usda``,
``scene.usdc`` or ``scene.usd``; a fetched release may instead be a record ``<id>.json`` whose
``usd_path`` points at the installed file (the last fetched release). A spec may pin a release's
``version``; another fetched release of the id is found by its own record,
``records/<commit>/<id>-<version>.json`` (a plain folder is unversioned and accepts any pin). No spec
ever holds a file path.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SCENES_ENV = "PHYGS_SCENES"
USER_CACHE = Path("~/.cache/phygs/scenes")
BUILTIN = Path(__file__).resolve().parent / "builtin"  # The scenes that ship with PhyGS.
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
    roots += [USER_CACHE.expanduser(), BUILTIN]
    return roots


def scene_file(scene_id: str, version: str | None = None) -> Path:
    """The scene's USD file (absolute); ``FileNotFoundError`` names the roots searched. A fetched
    release's record must carry ``version`` when one is given."""
    if not valid_id(scene_id):
        raise ValueError(f"Invalid scene id {scene_id!r}: one folder name, no path separators or '..'.")
    for root in scene_roots():
        folder = root / scene_id
        for name in FILES:
            if (folder / name).is_file():
                return (folder / name).resolve()
        record = root / f"{scene_id}.json"
        if record.is_file():
            data = json.loads(record.read_text())
            if version is not None and data.get("scene_version") != version:
                # Another fetched release of the id keeps its own record (the newest fetch first).
                fetched = sorted(
                    (root / "records").glob(f"*/{scene_id}-{version}.json"), key=lambda p: p.stat().st_mtime
                )
                if not fetched:
                    raise ValueError(
                        f"{record} holds {scene_id} {data.get('scene_version')}, the spec wants {version}: fetch it"
                        f" (phygs.scene.fetch fetch --scene_id {scene_id} --scene_version {version})."
                    )
                data = json.loads(fetched[-1].read_text())
            path = Path(data["usd_path"])
            if path.is_file():
                return path.resolve()
    searched = ", ".join(str(root) for root in scene_roots())
    raise FileNotFoundError(f"No scene {scene_id!r} under {searched}; set ${SCENES_ENV} or fetch the scene.")
