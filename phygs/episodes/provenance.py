# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What an episode ran on: content hashes of its assets and the code version, for its report."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def usd_sha256(path: str | Path) -> dict:
    """SHA-256 over a USD file and every layer it composes (sublayers, references, payloads), so an
    edit to any of them changes it (e.g. the robot's ``spot_physics.usd`` masses). Textures and other
    non-layer assets are left out: they do not change the simulation. A layer that is not a local
    file (a URL, such as the robot's camera models on the Isaac asset server) counts by its
    location: ``remote`` of them."""
    from pxr import UsdUtils

    layers, _, _ = UsdUtils.ComputeAllDependencies(str(path))
    return dict(path=str(Path(path).resolve()), **layers_sha256(layer.realPath or layer.identifier for layer in layers))


def layers_sha256(locations) -> dict:
    """SHA-256 over layer locations: a local file by its content, anything else (a URL) by location."""
    entries, remote = set(), 0
    for location in locations:
        file = Path(location)
        if "://" not in location and file.is_file():
            entries.add(f"{file.name}\0{file_sha256(file)}")
        else:
            entries.add(f"{location}\0remote")
            remote += 1
    digest = hashlib.sha256()
    for entry in sorted(entries):
        digest.update(entry.encode())
    return dict(sha256=digest.hexdigest(), layers=len(entries), remote=remote)


def code_version(root: str | Path) -> dict:
    """The git commit of ``root`` and whether its tracked files differ from it (``dirty``); None
    values outside a git checkout."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no") if commit else None
    return dict(commit=commit, dirty=None if status is None else bool(status))
