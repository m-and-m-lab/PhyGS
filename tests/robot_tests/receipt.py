# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run receipts for the robot tests: ``manifest.json`` in a folder, written by an external receipt script.

The script is a run-logging receipt script (``write_manifest.py``), found through the
``RP_WRITE_MANIFEST`` environment variable; the runner only calls it. :func:`start` records the code
(the PhyGS checkout's commit and, for a dirty tree, the patch; code the tests import from elsewhere
too), the interpreter and package versions, the tags and the command that reruns the folder;
:func:`finish` adds the end time and the exit code. Without the variable the tests run with one warning
and no receipt. A receipt that cannot be written or finished is a warning, never a test's failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ENV = "RP_WRITE_MANIFEST"  # The receipt script's path.
FILE = "manifest.json"
_warned = False


def script() -> Path | None:
    """The receipt script, or None when ``RP_WRITE_MANIFEST`` is unset or names no file."""
    path = os.environ.get(ENV)
    return Path(path) if path and Path(path).is_file() else None


def start(
    folder: Path,
    command: Sequence[str],
    *,
    repos: Sequence[Path],
    purpose: str,
    config: Path | None = None,
    tags: Sequence[str] = (),
) -> bool:
    """Write ``folder/manifest.json`` for ``command``; True when written, else a warning."""
    global _warned
    path = script()
    if path is None:
        if not _warned:
            print(f"run_suite: no run receipts ({ENV} is not set to write_manifest.py).", file=sys.stderr, flush=True)
            _warned = True
        return False
    call = [sys.executable, str(path), "--out", str(folder), "--repos", *map(str, repos)]
    call += [f"--purpose={purpose}", f"--python={sys.executable}"]
    call += [f"--config={config}"] if config is not None else []
    call += ["--tag", *tags] if tags else []
    try:
        done = subprocess.run([*call, "--", *map(str, command)], capture_output=True, text=True, timeout=600)
    except Exception as error:  # noqa: BLE001 - a receipt never stops the tests (module doc).
        print(f"run_suite: no run receipt in {folder}: {error}", file=sys.stderr, flush=True)
        return False
    if done.returncode or not (folder / FILE).is_file():
        print(f"run_suite: no run receipt in {folder}: {done.stderr.strip()[-300:]}", file=sys.stderr, flush=True)
        return False
    return True


def finish(folder: Path, code: int) -> None:
    """Add the end time and ``code`` to ``folder/manifest.json``; never raises."""
    path = script()
    if path is None or not (folder / FILE).is_file():
        return
    try:
        done = subprocess.run(
            [sys.executable, str(path), "--finish", str(folder), "--exit-status", str(int(code))],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if done.returncode:
            print(f"run_suite: could not finish {folder / FILE}: {done.stderr.strip()[-300:]}", file=sys.stderr)
    except Exception as error:  # noqa: BLE001 - a receipt never stops the tests (module doc).
        print(f"run_suite: could not finish {folder / FILE}: {error}", file=sys.stderr)
