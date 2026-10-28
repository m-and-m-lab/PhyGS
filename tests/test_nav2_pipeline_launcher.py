# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The dedicated launcher must select real Nav2 and matching experiments."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

path = Path(__file__).parents[1] / "phygs_robot_tests" / "components" / "check_nav2_pipeline.py"
spec = importlib.util.spec_from_file_location("nav2_pipeline_launcher", path)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_house_search_runs_on_native_pointcloud_servers():
    args = launcher.parse_args(["--output", "out"])
    assert args.ros_domain_id == 47
    launch, simulation = launcher.commands(args)
    assert launch[:2] == ["ros2", "launch"] and "sensor_mode:=pointcloud" in launch
    assert any(item.endswith("configs/house.yaml") for item in launch)
    assert simulation[1].endswith("components/check_house.py")
    assert simulation[simulation.index("--output") + 1].endswith("out/simulation")
    assert "--viz" not in simulation  # Headless by default.


def test_kit_window_and_video_on_request():
    simulation = launcher.commands(launcher.parse_args(["--viz", "kit", "--record", "--output", "out"]))[1]
    assert simulation[simulation.index("--viz") + 1] == "kit" and "--record" in simulation


@pytest.mark.parametrize(
    "arguments",
    [
        [],  # No output folder.
        ["--ros_domain_id", "-1"],
        ["--ros_domain_id", "102"],
        ["--input", "known_map"],  # Navigation is the agent test episodes/check_navigation.py.
    ],
)
def test_invalid_arguments_fail_before_launch(arguments):
    with pytest.raises(SystemExit):
        launcher.parse_args([*arguments, *(["--output", "out"] if arguments else [])])


def _launch_module():
    project = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "nav2_parameters", project / "phygs" / "execution" / "navigation" / "launch" / "nav2.launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return project, module


def test_nav2_plans_and_follows_the_padded_standing_rectangle():
    project, module = _launch_module()
    config = project / "phygs_robot_tests/configs/house.yaml"
    _, parameters = module.navigation_parameters(config, "pointcloud")
    profile = json.loads(
        (config.parent / yaml.safe_load(config.read_text())["transit"]["footprint_profile"]).read_text()
    )
    x0, x1, y0, y1 = profile["standing"]["bounds"]
    for name in ("global_costmap", "local_costmap"):
        costmap = parameters[name][name]["ros__parameters"]
        assert json.loads(costmap["footprint"]) == [[x1, y1], [x1, y0], [x0, y0], [x0, y1]]
        assert "robot_radius" not in costmap or costmap["footprint"] != "[]"
        padded = np.abs(np.asarray(json.loads(costmap["footprint"]))) + costmap["footprint_padding"]
        # The planner and the controller check the footprint against cost inflated past its far corner.
        assert float(np.linalg.norm(padded, axis=1).max()) < costmap["inflation_layer"]["inflation_radius"]
    planner = parameters["planner_server"]["ros__parameters"]["GridBased"]
    assert planner["plugin"] == "nav2_smac_planner::SmacPlannerLattice"
    lattice = json.loads(Path(planner["lattice_filepath"]).read_text())["lattice_metadata"]
    assert lattice["motion_model"] == "omni" and lattice["grid_resolution"] == costmap["resolution"]
    controller = parameters["controller_server"]["ros__parameters"]["FollowPath"]
    navigation = yaml.safe_load(config.read_text())["navigation"]
    assert controller["plugin"] == "nav2_mppi_controller::MPPIController" and controller["motion_model"] == "Omni"
    assert controller["CostCritic"]["consider_footprint"] is True
    assert (controller["vx_max"], controller["wz_max"]) == (navigation["max_speed"], navigation["max_yaw_speed"])


def test_lattice_primitives_are_scaled_to_the_costmap_resolution(tmp_path):
    _, module = _launch_module()
    source = tmp_path / "omni"
    source.mkdir()
    lattice = dict(
        lattice_metadata=dict(motion_model="omni", grid_resolution=0.05, turning_radius=0.5),
        primitives=[
            dict(
                trajectory_radius=0.5,
                trajectory_length=0.15,
                arc_length=0.0,
                straight_length=0.15,
                poses=[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.15, 0.0, 0.0]],
            ),  # fmt: skip
            dict(
                trajectory_radius=0.0,
                trajectory_length=0.0,
                arc_length=0.0,
                straight_length=0.0,
                poses=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.39]],
            ),  # A turn in place.  # fmt: skip
        ],
    )
    (source / "output.json").write_text(json.dumps(lattice))
    path = module.lattice_primitives(0.1, source)
    real = module.lattice_primitives(0.1)  # The installed lattice at the same resolution: its own file.
    assert path != real and "num_of_headings" in json.loads(real.read_text())["lattice_metadata"]
    scaled = json.loads(path.read_text())
    assert scaled["lattice_metadata"] == dict(motion_model="omni", grid_resolution=0.1, turning_radius=1.0)
    assert scaled["primitives"][0]["poses"][-1] == [0.3, 0.0, 0.0] and scaled["primitives"][0]["straight_length"] == 0.3
    assert scaled["primitives"][1]["poses"][-1] == [0.0, 0.0, 0.39]
    with pytest.raises(ValueError, match="multiple"):
        module.lattice_primitives(0.07, source)


def test_camera_clouds_feed_a_voxel_layer_that_clears_with_floor_returns():
    """Depth cameras feed Nav2's voxel layer: it marks points in the height band and clears along every
    ray in 3D, so the source keeps floor returns and returns above the band (they clear) while the layer
    marks only within the band. The 16-voxel column spans the floor to the band's top."""
    project, module = _launch_module()
    config = project / "phygs_robot_tests/configs/house.yaml"
    experiment = yaml.safe_load(config.read_text())
    low, high = experiment["mapping"]["height"]
    _, parameters = module.navigation_parameters(config, "pointcloud")
    for name in ("global_costmap", "local_costmap"):
        costmap = parameters[name][name]["ros__parameters"]
        assert costmap["plugins"] == ["static_layer", "obstacle_layer", "inflation_layer"]
        layer = costmap["obstacle_layer"]
        assert layer["plugin"] == "nav2_costmap_2d::VoxelLayer"
        assert layer["observation_sources"] == "inspection inspection_clearing"
        assert layer["origin_z"] == 0.0 and layer["z_voxels"] == 16
        assert layer["z_voxels"] * layer["z_resolution"] == pytest.approx(high)
        assert [layer["min_obstacle_height"], layer["max_obstacle_height"]] == [low, high]
        marking, clearing = layer["inspection"], layer["inspection_clearing"]
        assert marking["topic"] == "/spot_navigation/points/inspection" and marking["data_type"] == "PointCloud2"
        assert marking["marking"] and not marking["clearing"]
        assert clearing["topic"] == "/spot_navigation/clearing/inspection"
        assert clearing["clearing"] and not clearing["marking"]
        source = clearing
        # Floor returns and returns above the band clear; the layer marks only within the band.
        assert source["min_obstacle_height"] < 0.0 and source["max_obstacle_height"] > high
        assert source["obstacle_max_range"] == source["raytrace_max_range"] == experiment["mapping"]["max_depth"]
