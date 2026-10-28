# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the optional known-map Nav2 backend with the diagnostic's measured geometry."""

import copy
import json
import math
import shutil
from pathlib import Path

import yaml

# Launched servers get SIGTERM when the launcher dies (Linux parent-death signal, util-linux setpriv).
SERVER_PREFIX = "setpriv --pdeathsig TERM"


# Nav2's voxel grid holds at most 16 voxels per column (one 32-bit word with unknown bits).
VOXEL_COLUMN = 16
# Floor returns [m] (z near 0) reach the voxel layer for clearing only; it marks from the band's bottom.
FLOOR_SLACK = 0.1
# Returns above the band (walls, a ceiling) up to this high [m] clear too: their rays cross the band
# (the voxel layer clamps them at its top); the layer marks only up to the band's top.
CLEARING_CEILING = 10.0


def sensor_cameras(experiment: dict) -> list[str]:
    """The cameras whose point clouds feed Nav2's voxel layer."""
    return list(experiment["mapping"].get("cameras") or [experiment["mapping"]["camera"]])


def voxel_layer(experiment: dict, namespace: str) -> dict:
    """Nav2 ``VoxelLayer`` parameters: the bridge's camera clouds in a 3D grid from the floor to the top
    of the height band. Each point of ``clearing/<camera>`` (all the camera's points) clears the
    voxels along its ray from the camera, in 3D; each of ``points/<camera>`` (the same without the
    image's bottom margin) marks its voxel when it lies in the band. A column with a marked voxel is
    an obstacle cell. Rays to the floor and to walls above the band (in the source's band, outside the
    layer's) clear what has moved away; a ray over a box clears only the voxels above it, so the box's
    column stays marked."""
    low, high = (float(value) for value in experiment["mapping"]["height"])
    max_depth = float(experiment["mapping"]["max_depth"])
    cameras = sensor_cameras(experiment)
    return dict(
        plugin="nav2_costmap_2d::VoxelLayer",
        origin_z=0.0,
        z_voxels=VOXEL_COLUMN,
        z_resolution=high / VOXEL_COLUMN,
        min_obstacle_height=low,  # Marking: points in the band only.
        max_obstacle_height=high,
        mark_threshold=0,
        publish_voxel_map=False,
        observation_sources=" ".join(f"{camera} {camera}_clearing" for camera in cameras),
        **{
            source: dict(
                topic=f"/{namespace}/{kind}/{camera}",
                data_type="PointCloud2",
                marking=kind == "points",
                clearing=kind == "clearing",
                min_obstacle_height=-FLOOR_SLACK,  # Floor returns clear.
                max_obstacle_height=CLEARING_CEILING,  # So do returns above the band.
                obstacle_min_range=0.05,
                obstacle_max_range=max_depth,
                raytrace_min_range=0.05,
                raytrace_max_range=max_depth,
            )
            for camera in cameras
            for source, kind in ((camera, "points"), (f"{camera}_clearing", "clearing"))
        },
    )


def navigation_parameters(config_path: Path, sensor_mode: str) -> tuple[str, dict]:
    """Build matching planner, controller, and costmap parameters from an experiment."""
    experiment = yaml.safe_load(config_path.read_text())
    observed = "mapping" in experiment
    if sensor_mode not in {"occupancy_grid", "pointcloud"}:
        raise ValueError("Nav2 sensor_mode must be occupancy_grid or pointcloud")
    if (sensor_mode == "pointcloud") != observed:
        raise ValueError("Camera experiments (with mapping) run Nav2 on point clouds, known maps on occupancy grids")
    transit = experiment["transit"] if observed else experiment
    footprint_path = config_path.parent / transit["footprint_profile"]
    footprint = json.loads(footprint_path.read_text())
    if footprint["schema"] != 2 or footprint["status"] != "passed":
        raise ValueError("Nav2 requires a passed footprint calibration (schema 2)")
    namespace = experiment["nav2"]["namespace"]
    parameters_path = Path(__file__).resolve().parents[3] / "configs" / "navigation" / "nav2.yaml"
    parameters = yaml.safe_load(parameters_path.read_text())
    geometry = {**experiment["mapping"], **transit} if observed else experiment["map"]
    limits = experiment["navigation"]
    # Nav2 plans the measured standing outline, padded as the built-in planner pads it.
    outline, padding = standing_outline(footprint["standing"]), footprint["standing"]["padding"] + geometry["clearance"]
    radius = circumscribed_radius(footprint["standing"], padding)
    decay_distance = experiment["nav2"].get("inflation_cost_distance", geometry.get("clearance_cost_distance", 0.3))
    if not isinstance(decay_distance, (int, float)) or not 0 < decay_distance <= 1.0:
        raise ValueError("Nav2 inflation cost distance must be in (0, 1] m")
    for name in ("global_costmap", "local_costmap"):
        costmap = copy.deepcopy(parameters[name][name]["ros__parameters"])
        costmap.update(
            footprint=json.dumps(outline),
            footprint_padding=padding,
            resolution=geometry["resolution"],
            global_frame=namespace + "/map",
            robot_base_frame=namespace + "/base_link",
        )
        costmap["inflation_layer"]["inflate_around_unknown"] = False
        if name == "local_costmap":
            # MPPI's costmap: a window that follows the robot, as usual in Nav2. The global costmap
            # covers the map; a full-map local one raytraced every camera point a second time.
            costmap.update(rolling_window=True, width=LOCAL_WINDOW, height=LOCAL_WINDOW)
        if sensor_mode == "pointcloud":
            # 3D clearing, the standard for depth cameras: Nav2 keeps a source's points only within its
            # height band, so a 2D obstacle layer fed the clouds never cleared with floor returns (a moved
            # box stayed), and a 2D floor clearing source wore away a box's edges (rays over the box; the
            # robot touched it, 2026-10-25).
            costmap["plugins"] = ["static_layer", "obstacle_layer", "inflation_layer"]
            costmap["obstacle_layer"] = voxel_layer(experiment, namespace)
        costmap["static_layer"]["map_topic"] = "/" + namespace + "/map"
        costmap["inflation_layer"]["inflation_radius"] = radius + 3 * decay_distance
        costmap["inflation_layer"]["cost_scaling_factor"] = 1 / decay_distance
        parameters[name][name]["ros__parameters"] = costmap
    planner = parameters["planner_server"]["ros__parameters"]["GridBased"]
    # The base's kinematics pick MPPI's motion model and the lattice's moves.
    motion_model, primitives = MOTION_MODELS[experiment["base_kinematics"]]
    planner["lattice_filepath"] = str(lattice_primitives(geometry["resolution"], SAMPLE_PRIMITIVES / primitives))
    controller = parameters["controller_server"]["ros__parameters"]
    controller["controller_frequency"] = 1 / limits["period"]
    # MPPI predicts each sample with one command per control period over a fixed horizon.
    period = limits["period"]
    controller["FollowPath"].update(
        motion_model=motion_model,
        vx_max=limits["max_speed"],
        wz_max=limits["max_yaw_speed"],
        model_dt=period,
        time_steps=round(MPPI_HORIZON_SECONDS / period),
    )
    return namespace, parameters


MPPI_HORIZON_SECONDS = 2.0  # How far ahead MPPI scores its sampled commands [s].
# Base kinematics (robot.yaml) -> MPPI motion model and the installed Smac lattice set (in SAMPLE_PRIMITIVES).
MOTION_MODELS = {
    "holonomic": ("Omni", "0.5m_turning_radius/omni"),
    "differential": ("DiffDrive", "0.5m_turning_radius/diff"),
}
LOCAL_WINDOW = 6  # The local costmap's side [m]: 2 s of walking at 1 m/s either way, and the footprint.


def standing_outline(standing: dict) -> list[list[float]]:
    """Corners [m] of the measured standing outline in the base frame, as a Nav2 ``footprint``."""
    x0, x1, y0, y1 = standing["bounds"]
    return [[x1, y1], [x1, y0], [x0, y0], [x0, y1]]


def circumscribed_radius(standing: dict, padding: float) -> float:
    """Radius [m] of the circle about the base enclosing the standing outline padded by ``padding``."""
    x0, x1, y0, y1 = standing["bounds"]
    return math.hypot(max(-x0, x1) + padding, max(-y0, y1) + padding)


SAMPLE_PRIMITIVES = Path("/opt/ros/jazzy/share/nav2_smac_planner/sample_primitives/5cm_resolution")


def lattice_primitives(resolution: float, source: Path = SAMPLE_PRIMITIVES / "0.5m_turning_radius/omni") -> Path:
    """Smac Lattice motion primitives [m] for ``resolution`` [m]: the installed 5 cm set, scaled.

    Scaling a lattice keeps its end poses on the grid and its headings; the turning radius and the
    lengths scale with it. Returns the written file (in the temporary directory). The file is named
    by its content and replaced atomically: a planner starting meanwhile never reads another
    lattice or a half-written file (a unit test's small lattice once replaced the 10 cm one under a
    starting planner, 2026-10-25).
    """
    import hashlib
    import os
    import tempfile

    lattice = json.loads((source / "output.json").read_text())
    metadata = lattice["lattice_metadata"]
    scale = resolution / metadata["grid_resolution"]
    if scale < 1 or not math.isclose(scale, round(scale)):
        raise ValueError(f"Lattice resolution {resolution} m must be a multiple of {metadata['grid_resolution']} m")
    metadata["grid_resolution"] = resolution
    metadata["turning_radius"] *= scale
    for primitive in lattice["primitives"]:
        for key in ("trajectory_radius", "trajectory_length", "arc_length", "straight_length"):
            primitive[key] *= scale
        primitive["poses"] = [[x * scale, y * scale, yaw] for x, y, yaw in primitive["poses"]]
    text = json.dumps(lattice)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    name = f"nav2_lattice_{metadata['motion_model']}_{round(resolution * 1000)}mm_{digest}.json"
    path = Path(tempfile.gettempdir()) / name
    staging = path.with_suffix(f".{os.getpid()}.tmp")
    staging.write_text(text)
    os.replace(staging, path)
    return path


def launch_nodes(context):
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node

    config_path = Path(LaunchConfiguration("experiment").perform(context)).resolve()
    sensor_mode = LaunchConfiguration("sensor_mode").perform(context)
    namespace, parameters = navigation_parameters(config_path, sensor_mode)

    # ParameterFile handles the nested costmap nodes without requiring a ROS package installation.
    import tempfile

    from launch_ros.descriptions import ParameterFile
    from nav2_common.launch import RewrittenYaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(parameters, stream)
        parameter_path = stream.name
    configured = ParameterFile(
        RewrittenYaml(source_file=parameter_path, root_key=namespace, param_rewrites={}, convert_types=True),
        allow_substs=True,
    )
    # Each server stops when this launcher dies, so a killed run leaves no Nav2 on its ROS domain.
    prefix = SERVER_PREFIX if shutil.which("setpriv") else None
    nodes = [
        Node(
            package=package,
            executable=name,
            name=name,
            namespace=namespace,
            parameters=[configured],
            output="screen",
            prefix=prefix,
        )
        for package, name in (("nav2_planner", "planner_server"), ("nav2_controller", "controller_server"))
    ]
    nodes.append(
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            namespace=namespace,
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[dict(use_sim_time=True, autostart=True, node_names=["planner_server", "controller_server"])],
            prefix=prefix,
        )
    )
    return nodes


def generate_launch_description():
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    return LaunchDescription(
        [
            DeclareLaunchArgument("experiment", description="The experiment YAML, e.g. a run's nav2_experiment.yaml"),
            DeclareLaunchArgument("sensor_mode", default_value="occupancy_grid"),
            OpaqueFunction(function=launch_nodes),
        ]
    )
