# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public skill APIs load their execution package without importing manipulation or ROS."""

import os
import subprocess
import sys
from pathlib import Path


def test_skill_apis_import_without_manipulation_or_ros(tmp_path):
    project = Path(__file__).resolve().parents[1]
    source = project
    dependency_paths = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(project)]
    code = """
import sys
from pathlib import Path
from phygs.skills.locomotion.api import LocomotionClient, CommandFeedback
from phygs.skills.navigation.api import (
    NavigationClient, NavigationGoal, NavigationConfig, NavigationFeedback,
)
from phygs.observations.types import ObservationConfig
from phygs.robots.control import RobotControl, base_controller_class
from phygs.robots.description import load_robot
robot = load_robot("spot")
robot.validate_urdf()
files = base_controller_class(robot).files(robot)  # The walking policy and its training configuration.
assert all(path.is_file() for path in files.values())
assert LocomotionClient.__module__ == 'phygs.skills.locomotion.api'
assert NavigationClient.__module__ == 'phygs.skills.navigation.api'
assert not any(name.startswith("skills.manipulation") for name in sys.modules)
assert files["policy_sha256"].name == "policy.pt" and files["policy_config_sha256"].name == "env.yaml"
from phygs.skills.manipulation.api import ManipulationClient
assert ManipulationClient.__module__ == 'phygs.skills.manipulation.api'
assert not any(name.startswith(('open3d', 'curobo')) for name in sys.modules)
assert 'rclpy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(source), *dependency_paths])},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
