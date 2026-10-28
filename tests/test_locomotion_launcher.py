# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check launch-mode selection before starting Isaac Sim."""

import subprocess
import sys
from pathlib import Path

LOCOMOTION = Path(__file__).parents[1] / "phygs_robot_tests" / "components" / "check_locomotion.py"


def test_an_output_folder_is_required(tmp_path):
    result = subprocess.run(
        [sys.executable, str(LOCOMOTION), "--config", str(tmp_path / "missing.yaml")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--output" in result.stderr


def test_automatic_mode_is_the_default_and_reads_the_config_before_starting(tmp_path):
    missing = tmp_path / "missing.yaml"
    result = subprocess.run(
        [sys.executable, str(LOCOMOTION), "--config", str(missing), "--output", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FileNotFoundError" in result.stderr and str(missing) in result.stderr
    assert not (tmp_path / "out").exists()  # Nothing started: the config is read first.
