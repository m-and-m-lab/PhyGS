# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bear house under Nav2: search for a magenta cube with RGB-D while a colour detector watches the
camera: ``navigation.explore()`` (explore_lite on Nav2's costmap of the unexplored kitchen), then,
when the map is complete but the cube unseen, a coverage pass (viewpoints spread over the free floor,
a turn at each). Run it through ``check_nav2_pipeline.py``, which starts the ROS servers. Navigation
in the house is the agent test ``episodes/check_navigation.py`` (spec ``nav_house``).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # The PhyGS checkout, without pip install.
from phygs_robot_tests import common  # noqa: E402

flags = common.parser(__doc__)
flags.add_argument("--config", type=Path, default=common.CONFIGS / "house.yaml")
flags.set_defaults(enable_cameras=True)
args = flags.parse_args()
config = yaml.safe_load(args.config.read_text())
from phygs.episodes.scenes import scene_file  # noqa: E402

config["scene"]["usd_path"] = str(scene_file(config["scene"]["id"]))
check = common.Check("check_house search", args)
robot_description = common.robot(args)  # --robot (default spot).

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from PIL import Image  # noqa: E402

from phygs.execution.navigation.backends.nav2 import Nav2BackendFactory, Nav2Config  # noqa: E402
from phygs.execution.navigation.floor import DeclaredFloorMap  # noqa: E402
from phygs.execution.navigation.footprint import (  # noqa: E402
    load_footprint,
    transit_footprint,
    validate_transit,
)
from phygs.observations.types import (  # noqa: E402
    CameraPermission,
    ObservationAccess,
    ObservationConfig,
)
from phygs.robots.control import RobotControl, base_controller_class  # noqa: E402
from phygs.robots.scene import make_scene_cfg, simulation_cfg  # noqa: E402
from phygs.runtime.episode import RunMode  # noqa: E402
from phygs.runtime.pipeline import PipelineConfig  # noqa: E402
from phygs.runtime.timing import duration_ticks  # noqa: E402
from phygs.simulation.scene_assets import (  # noqa: E402
    SceneBodies,
    register_scene_bodies,
    validate_spawn_disk,
)
from phygs.simulation.world import IsaacLabWorld  # noqa: E402
from phygs.skills.navigation.api import NavigationConfig  # noqa: E402
from phygs.tools.color_detection import magenta_pixel  # noqa: E402
from phygs_robot_tests.tasks.object_search import ObjectSearchEvaluator, ObjectSearchTask, TargetBounds  # noqa: E402
from phygs_robot_tests.tasks.object_search_api import ObjectSearchClient  # noqa: E402

require = common.require


def create_scene(
    config: dict, *, device: str, dt: float
) -> tuple[sim_utils.SimulationContext, InteractiveScene, tuple[str, ...]]:
    """Load the house and register its bodies before spawning the robot and the search target."""
    stage = sim_utils.open_stage(config["scene"]["usd_path"])
    validate_spawn_disk(
        stage,
        tuple(config["robot"]["position"][:2]),
        config["transit"]["initial_free_radius"],
        tuple(config["mapping"]["height"]),
    )
    sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, render_interval=16, device=device))
    cfg = make_scene_cfg(**common.config_scene(robot_description, config))(
        num_envs=1, env_spacing=5, replicate_physics=False
    )
    cfg.ground = None
    body_names = register_scene_bodies(stage, cfg)
    target = config["search"]["target"]
    cfg.search_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SearchTarget",
        spawn=sim_utils.CuboidCfg(
            size=tuple(target["size"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(target["color"])),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(target["center"])),
    )
    scene = InteractiveScene(cfg)
    sim.reset()
    scene.update(dt)
    x, y, z = config["robot"]["position"]
    sim.set_camera_view(eye=(x + 2, y - 3, z + 2), target=(x, y, z))
    return sim, scene, body_names


def create_control(
    config: dict,
    config_path: Path,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    dt: float,
    *,
    output: Path,
) -> tuple[RobotControl, ObservationConfig, float]:
    """Assemble ground-truth observations and native Nav2 from the experiment config. Nav2's map is
    the kitchen (``exploration.bounds``), unexplored beyond the spawn disk."""
    footprint = load_footprint(
        (config_path.parent / config["transit"]["footprint_profile"]).resolve(strict=True),
        robot_description.usd,
        base_controller_class(robot_description).files(robot_description),
    )
    navigation = NavigationConfig(**config["navigation"], arm_posture=robot_description.arm.transit_posture)
    validate_transit(footprint, navigation, dt)
    capture = ObservationConfig(
        **{
            **config["observations"],
            "cameras": tuple(CameraPermission(**item) for item in config["observations"]["cameras"]),
        }
    )
    require(capture.camera_period <= 1 / 30, "Camera capture must be at least 30 simulated FPS")
    access = ObservationAccess.from_config(capture)
    rectangle = transit_footprint(footprint)
    x, y, _ = config["robot"]["position"]
    floor = DeclaredFloorMap(
        bounds=tuple(config["exploration"]["bounds"]),
        resolution=config["mapping"]["resolution"],
        footprint=rectangle,
        clearance=0,
        explored=(x, y, config["transit"]["initial_free_radius"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    backend = Nav2BackendFactory(
        Nav2Config(
            **config["nav2"],
            sensor_cameras=(config["mapping"]["camera"],),
            sensor_period=capture.camera_period,
            explore_log=str(output / "explore.log"),
        ),
        maps=floor,
    )
    control = RobotControl.create(
        sim,
        scene,
        robot_description,
        mode=RunMode.AUTONOMOUS,
        pipeline_config=PipelineConfig(capture, access, access),
        navigation_config=navigation,
        navigation_backend_factory=backend,
        allowed_actions=("navigation", "motion"),
    )
    return control, capture, rectangle


def body_states(scene: InteractiveScene, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Read registered scene bodies' link poses and CoM velocities, by USD path, for exact reset comparisons."""
    bodies = SceneBodies(scene)
    return {name: torch.cat((bodies.pose(name), bodies.velocity(name))) for name in names}


def check_body_reset(world: IsaacLabWorld, scene: InteractiveScene, initial: dict[str, torch.Tensor]) -> None:
    """Check the world adapter directly, before episode settling advances physics."""
    world.restore()
    restored = body_states(scene, tuple(initial))
    require(
        all(torch.allclose(restored[name], state, atol=1e-5, rtol=0) for name, state in initial.items()),
        "House body states leaked across resets",
    )


def save_costmap(control: RobotControl, folder: Path) -> None:
    """Nav2's global costmap as ``costmap.npy`` (cost per cell) and ``costmap.png`` (+Y up): unknown
    grey, obstacles black, inflation shaded, free white."""
    costmap = control.navigation.backend.costmap()
    if costmap is None:
        return
    costs = costmap[0]
    np.save(folder / "costmap.npy", costs)
    image = np.where(costs == 255, 128, 255 - np.minimum(costs, 254).astype(np.int32)).astype(np.uint8)
    Image.fromarray(np.flipud(image)).save(folder / "costmap.png")


def next_viewpoint(costmap, visited: list[tuple[float, float]], max_cost: int, separation: float, margin: float):
    """The coverage pass's next viewpoint [m]: the cell of Nav2's costmap ((costs, origin, cell size))
    at most ``max_cost``, at least ``margin`` [m] inside its edges (the robot fits), farthest from
    every ``visited`` point, or None when none is ``separation`` [m] away. Exploration has mapped the
    floor; the viewpoints put the camera where it has not been."""
    costs, origin, resolution = costmap
    edge = math.ceil(margin / resolution)
    inner = np.zeros(costs.shape, dtype=bool)
    inner[edge:-edge, edge:-edge] = True
    rows, cols = np.nonzero((costs <= max_cost) & inner)
    if not len(rows):
        return None
    points = np.stack((origin[0] + (cols + 0.5) * resolution, origin[1] + (rows + 0.5) * resolution), axis=1)
    distance = np.min(np.linalg.norm(points[:, None] - np.asarray(visited)[None], axis=2), axis=1)
    best = int(np.argmax(distance))
    return None if distance[best] < separation else tuple(float(value) for value in points[best])


def run(report: dict) -> None:
    dt = common.PHYSICS_DT
    sim, scene, body_names = create_scene(config, device=args.device, dt=dt)
    control, capture, _ = create_control(config, args.config, sim, scene, dt, output=args.output)
    check.record(control, scene["robot"])
    # This fixture exercises world restoration directly, without reaching into runtime internals.
    reset_world = IsaacLabWorld(sim, scene)
    initial = body_states(scene, body_names)
    target = config["search"]["target"]
    settings = config["house"]
    report.update(
        scene=config["scene"]["usd_path"],
        registered_bodies=len(body_names),
        episodes=[],
        localization="ground_truth",
        camera_period_sim_seconds=capture.camera_period,
        navigation_information="declared_flat_floor_and_camera_pointclouds",
        exploration_bounds=config["exploration"]["bounds"],
    )
    camera_ticks = duration_ticks(capture.camera_period, dt)
    frames = []
    exploring = None  # The exploration command while it runs.
    exploration = []  # The exploration command's distinct feedback reasons, in order.
    evaluator = None
    client = None
    found = False

    def inspect_frame() -> None:
        nonlocal found
        packet = control.observe()
        camera = packet.cameras[config["mapping"]["camera"]]
        require(tuple(camera.depth.shape) == (480, 640), "Depth capture is not 640x480")
        if not frames or frames[-1] != camera.stamp.time_seconds:
            frames.append(camera.stamp.time_seconds)
        center = scene["search_target"].data.root_pos_w.torch[0].tolist()
        half = np.asarray(target["size"]) / 2
        evaluator.record(packet, TargetBounds(tuple(np.asarray(center) - half), tuple(np.asarray(center) + half)))
        if not found and evaluator.status == "running":
            pixel = magenta_pixel(camera)
            if pixel is not None:
                result = client.declare_target(camera.stamp, *pixel)
                report["search"] = evaluator.result()
                require(result == "succeeded", "Reference detector submitted invalid target evidence")
                Image.fromarray(camera.rgb.cpu().numpy()).save(args.output / "found_rgb.png")
                found = True

    def step(seconds: float) -> None:
        for _ in range(duration_ticks(seconds, dt)):
            require(check.app.is_running(), "Simulation closed before test completed")
            control.step()
            if evaluator is not None:
                evaluator.update(control.runtime.time_seconds)
            if control.runtime.tick % duration_ticks(1.0, dt) == 0:
                pose = scene["robot"].data.root_pose_w.torch[0].tolist()
                require(pose[2] > 0.3 and 1 - 2 * (pose[3] ** 2 + pose[4] ** 2) > 0.5, "Spot fell")
                print(json.dumps({"sim_seconds": control.runtime.time_seconds, "position": pose[:3]}), flush=True)
            if control.runtime.tick % camera_ticks == 0:
                inspect_frame()
            if exploring is not None:
                reason = control.navigation.feedback(exploring).reason
                if not exploration or exploration[-1] != reason:
                    exploration.append(reason)
                    print(json.dumps({"exploration": reason}), flush=True)

    def finish_command(command: int) -> dict:
        stopping = False
        while control.navigation.active:
            step(capture.camera_period)
            if (
                evaluator is not None
                and (found or evaluator.status != "running")
                and control.navigation.active
                and not stopping
            ):
                control.navigation.cancel(command)
                stopping = True
        feedback = control.navigation.feedback(command)
        pose = scene["robot"].data.root_pose_w.torch[0].tolist()
        speed = float(torch.linalg.vector_norm(scene["robot"].data.root_lin_vel_b.torch[0, :2]))
        yaw_speed = abs(float(scene["robot"].data.root_ang_vel_b.torch[0, 2]))
        record = dict(feedback=asdict(feedback), pose_xyzw=pose, speed_mps=speed, yaw_rate_radps=yaw_speed)
        print(json.dumps(record), flush=True)
        require(speed < 0.15 and yaw_speed < 0.2, "Navigation ended without physical stopping")
        return record

    try:
        check_body_reset(reset_world, scene, initial)
        control.reset(settling_seconds=settings["settling_seconds"])
        frames.clear()
        task = ObjectSearchTask(
            "bear_house",
            report["experiment_sha256"],
            config["search"]["target_description"],
            config["mapping"]["camera"],
            config["exploration"]["time_budget"],
        )
        evaluator = ObjectSearchEvaluator(task, episode_id=control.runtime.episode_id)
        client = ObjectSearchClient(evaluator.submit, camera=task.camera)
        started = time.monotonic()
        inspect_frame()
        control.locomotion.command_velocity(0, 0, settings["scan_yaw_radps"], duration=settings["scan_seconds"])
        step(settings["scan_seconds"])
        control.locomotion.stop()
        step(1.0)
        records = []
        report["current_navigation"] = records
        exploring = control.navigation.explore(timeout=config["exploration"]["time_budget"])
        records.append(finish_command(exploring))
        exploring = None
        report["exploration"] = exploration
        save_costmap(control, args.output)
        report["found_in"] = "exploration" if found else None
        coverage = config["exploration"]["coverage"]
        visited = [tuple(control.observe().localization.pose.matrix[:2, 3].tolist())]
        report["viewpoints"] = []
        while not found and evaluator.status == "running" and len(visited) <= coverage["max_viewpoints"]:
            view = next_viewpoint(
                control.navigation.backend.costmap(),
                visited,
                coverage["max_cost"],
                coverage["separation"],
                coverage["edge_margin"],
            )
            if view is None:
                report["coverage_outcome"] = "viewpoints_exhausted"
                break
            visited.append(view)
            print(json.dumps({"viewpoint": view}), flush=True)
            record = finish_command(
                control.navigation.navigate_to(*view, position_tolerance=coverage["arrival_tolerance"])
            )
            records.append(record)
            report["viewpoints"].append(dict(position=view, status=record["feedback"]["status"]))
            if found:
                report["found_in"] = f"coverage, walking to viewpoint {len(report['viewpoints'])}"
            if record["feedback"]["status"] != "succeeded" or found:
                continue
            turn = control.locomotion.command_velocity(
                0, 0, settings["scan_yaw_radps"], duration=coverage["turn_seconds"]
            )
            while not found and control.locomotion.feedback(turn).status in {"active", "stopping"}:
                step(capture.camera_period)
            control.locomotion.stop()
            step(1.0)
            if found:
                report["found_in"] = f"coverage, turning at viewpoint {len(report['viewpoints'])}"
        report["search"] = evaluator.result()
        save_costmap(control, args.output)
        require(found, "Search ended without grounded target evidence")
        require(
            len(frames) > 1 and np.max(np.diff(frames)) <= capture.camera_period + 1e-8,
            "Camera stream skipped simulated capture intervals",
        )
        sim_seconds = control.runtime.time_seconds
        result = dict(
            episode_id=control.runtime.episode_id,
            navigation=records,
            camera_fps_sim=(len(frames) - 1) / (frames[-1] - frames[0]),
            real_time_factor=sim_seconds / (time.monotonic() - started),
        )
        report["episodes"].append(result)
        print(json.dumps(result), flush=True)
        control.runtime.complete("House diagnostic completed")
    finally:
        if len(frames) > 1:
            report["camera_fps_sim"] = (len(frames) - 1) / (frames[-1] - frames[0])
            report["camera_max_gap_sim_seconds"] = float(np.max(np.diff(frames)))
        control.close()


check.data.update(
    backend="nav2",
    task="search",
    experiment_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
)
check.run(lambda: run(check.data))
