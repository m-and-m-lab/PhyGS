# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations: RGB-D alignment against a measured cuboid, the camera rate, an occluder launched
to reveal the background, permissions (a restricted view leaks nothing), and reset. ``--manual``
shows the cameras live while the occluder is launched every 4 s.

    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_observations.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_observations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # The PhyGS checkout, without pip install.
from phygs_robot_tests import common  # noqa: E402

flags = common.parser(__doc__, manual=True)
flags.add_argument("--preset", choices=("observations", "localized"), default="localized")
flags.add_argument("--config", type=Path, default=common.CONFIGS / "observations.yaml")
flags.set_defaults(enable_cameras=True)
args = flags.parse_args()
config = yaml.safe_load(args.config.read_text())  # The test's settings and its inspection camera.
check = common.Check(f"check_observations {args.preset}", args)
robot_description = common.robot(args)  # --robot (default spot).
# The robot as episodes build it, with its first front camera and the IMU, plus the inspection camera.
scene_arguments = common.robot_scene(robot_description, cameras=robot_description.cameras_for("front")[:1], imu=True)
scene_arguments["cameras"].update(config.pop("sensors")["cameras"])

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from PIL import Image  # noqa: E402

from phygs.observations.types import CameraPermission, ObservationConfig  # noqa: E402
from phygs.robots.control import RobotControl  # noqa: E402
from phygs.robots.scene import make_scene_cfg, simulation_cfg  # noqa: E402
from phygs.runtime.episode import RunMode  # noqa: E402
from phygs.runtime.pipeline import PipelineConfig  # noqa: E402
from phygs.runtime.timing import duration_ticks  # noqa: E402
from phygs.simulation.base import BaseLink  # noqa: E402
from phygs.simulation.observations import IsaacLabObservations, pose_matrix  # noqa: E402

require = common.require


def add_boxes(cfg, diagnostic: dict) -> None:
    """Add the static wall and the movable 1 kg obstacle box (``diagnostic`` positions and sizes [m])."""
    cfg.wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=tuple(diagnostic["wall_size"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.8)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(diagnostic["wall_position"])),
    )
    cfg.obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=tuple(diagnostic["obstacle_size"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(diagnostic["obstacle_position"])),
    )


def save_images(packet, label):
    for name, camera in packet.cameras.items():
        stem = args.output / f"{label}_{name}"
        if camera.rgb is not None:
            Image.fromarray(camera.rgb.cpu().numpy()).save(stem.with_suffix(".png"))
        if camera.depth is not None:
            np.save(str(stem) + "_depth.npy", camera.depth.cpu().numpy())
            # Preview only; metric depth and validity remain unchanged in the public packet.
            preview = torch.where(camera.depth_valid, (camera.depth / 5).clamp(0, 1), 0)
            Image.fromarray((preview.cpu().numpy() * 255).astype(np.uint8)).save(str(stem) + "_depth.png")


def describe(packet):
    return {
        "episode": packet.stamp.episode_id,
        "tick": packet.stamp.tick,
        "time_s": packet.stamp.time_seconds,
        "joint_names": packet.joints.names,
        "joint_positions_rad": packet.joints.positions.tolist(),
        "imu_gyro_radps": packet.imu.angular_velocity.tolist(),
        "imu_acceleration_mps2": packet.imu.proper_acceleration.tolist(),
        "localization_mode": packet.localization.mode if packet.localization else "none",
        "cameras": {
            name: {
                "capture_tick": camera.stamp.tick,
                "frame": camera.frame,
                "rgb_shape": list(camera.rgb.shape),
                "valid_depth_pixels": int(camera.depth_valid.sum()),
                "intrinsics": camera.intrinsics.tolist(),
                "body_from_optical": camera.extrinsics.matrix.tolist(),
                "center_depth_m": float(camera.depth[camera.depth.shape[0] // 2, camera.depth.shape[1] // 2]),
            }
            for name, camera in packet.cameras.items()
        },
    }


class CameraViewer:
    """Live inspection only; visualization never changes metric observations."""

    def __init__(self, packet):
        import omni.ui as ui

        self.window = ui.Window("Spot observations — RGB and depth (0–5 m)", width=680, height=580)
        self.providers = {}
        with self.window.frame:
            with ui.VStack():
                for name in packet.cameras:
                    ui.Label(name, height=20)
                    with ui.HStack(height=240):
                        for product in ("rgb", "depth"):
                            provider = ui.ByteImageProvider()
                            self.providers[name, product] = provider
                            ui.ImageWithProvider(provider, width=320, height=240)
        self.update(packet)

    def update(self, packet):
        for name, camera in packet.cameras.items():
            depth = torch.where(camera.depth_valid, (camera.depth / 5).clamp(0, 1), 0)
            gray = (depth.cpu().numpy() * 255).astype(np.uint8)
            for product, pixels in (
                ("rgb", camera.rgb.cpu().numpy()),
                ("depth", np.repeat(gray[..., None], 3, axis=2)),
            ):
                height, width = pixels.shape[:2]
                rgba = np.dstack((pixels, np.full((height, width), 255, dtype=np.uint8)))
                self.providers[name, product].set_bytes_data(rgba.flatten().data, [width, height])


def run(report):
    diagnostic = config["diagnostic"]
    dt = common.PHYSICS_DT
    sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, render_interval=16, device=args.device))
    sim.set_camera_view(eye=(4, 4, 3), target=(1, 0, 0.6))
    cfg = make_scene_cfg(**scene_arguments)(num_envs=1, env_spacing=5, replicate_physics=False)
    add_boxes(cfg, diagnostic)
    scene = InteractiveScene(cfg)
    sim.reset()
    scene.update(dt)
    permissions = dict(config["observations"])
    permissions["cameras"] = tuple(CameraPermission(**entry) for entry in permissions["cameras"])
    observation_config = ObservationConfig(**permissions)
    pipeline_config = (
        PipelineConfig.observations_only(observation_config)
        if args.preset == "observations"
        else PipelineConfig.localized(observation_config)
    )
    observation_config = pipeline_config.capture
    control = RobotControl.create(
        sim,
        scene,
        robot_description,
        mode=RunMode.HUMAN if args.manual else RunMode.AUTONOMOUS,
        pipeline_config=pipeline_config,
    )
    check.record(control, scene["robot"])
    torch.manual_seed(diagnostic["seed"])
    control.reset(settling_seconds=diagnostic["settling_seconds"])
    initial = control.observe()
    report.update(configuration=config, initial=describe(initial))
    print(json.dumps({"initial": report["initial"]}), flush=True)
    save_images(initial, "initial")
    viewer = CameraViewer(initial) if args.visualizer and "kit" in args.visualizer else None
    require(initial.joints.valid and initial.imu.valid, "Invalid robot observations")
    require(all(camera.valid and camera.depth_valid.any() for camera in initial.cameras.values()), "No valid RGB-D")
    require(control.observe().stamp == initial.stamp, "Reading observations advanced time")
    if args.manual:
        print("Inspect the scene and saved RGB/depth previews in " + str(args.output), flush=True)
        print(
            "The red obstacle is launched upward every 4 s; the camera observes its physical motion.",
            flush=True,
        )
        while check.app.is_running():
            if control.runtime.tick % duration_ticks(4.0, dt) == 0:
                launch_obstacle(scene, diagnostic)
            control.step()
            if control.runtime.tick % duration_ticks(observation_config.camera_period, dt) == 0:
                viewer.update(control.observe())
            if control.runtime.tick % duration_ticks(1.0, dt) == 0:
                packet = control.observe()
                print(json.dumps(describe(packet)), flush=True)
                save_images(packet, "latest")
        return "manual_finished"
    require(initial.stamp.tick == 0, "Reset did not restart observation timestamps")
    camera = initial.cameras["inspection"]
    y, x = camera.depth.shape[0] // 2, camera.depth.shape[1] // 2
    initial_depth = float(camera.depth[y, x])
    require(camera.depth_valid[y, x], "Occluder center depth is invalid")
    # Check the center pixel against the physically measured front face of the cuboid.
    pixel = torch.tensor([x, y, 1.0], device=camera.depth.device, dtype=camera.depth.dtype)
    optical_point = torch.linalg.solve(camera.intrinsics, pixel) * camera.depth[y, x]
    require(control.navigation is None, "Observation presets unexpectedly constructed navigation")
    require(not control.pipeline.configuration["services"], "Observation presets unexpectedly constructed processing")
    if args.preset == "observations":
        require(initial.localization is None, "Raw preset exposed localization")
        require(all(edge.target == "body" for edge in initial.transforms), "Raw preset exposed a global transform")
        # Privileged physical evaluation only; never put this pose in the agent packet.
        physical_pose = scene["robot"].data.root_pose_w.torch[0]
        world_from_camera = pose_matrix(physical_pose[:3], physical_pose[3:]) @ camera.extrinsics.matrix
    else:
        world_from_camera = initial.transform("world", camera.frame, camera.stamp)
    world_point = world_from_camera[:3, :3] @ optical_point + world_from_camera[:3, 3]
    expected_x = float(scene["obstacle"].data.root_pos_w.torch[0, 0]) - diagnostic["obstacle_size"][0] / 2
    plane_error = abs(float(world_point[0]) - expected_x)
    require(plane_error < diagnostic["plane_tolerance_m"], f"Depth scale/frame mismatch: {plane_error} m")
    points = camera.points()
    require(points.shape == (int(camera.depth_valid.sum()), 3), "Point reconstruction lost pixels")
    control.step()
    held = control.observe()
    require(held.cameras["inspection"].stamp == camera.stamp, "Camera rate ignored")
    require(torch.equal(held.cameras["inspection"].depth, camera.depth), "Image changed between captures")
    launch_obstacle(scene, diagnostic)
    for _ in range(duration_ticks(diagnostic["lift_seconds"], dt) - 1):
        control.step()
    changed = control.observe()
    changed_camera = changed.cameras["inspection"]
    changed_depth = float(changed_camera.depth[y, x])
    report.update(
        initial_depth_m=initial_depth,
        revealed_depth_m=changed_depth,
        obstacle_pose_after_launch=scene["obstacle"].data.root_pose_w.torch[0].tolist(),
    )
    save_images(changed, "revealed")
    require(
        changed_depth - initial_depth > diagnostic["minimum_depth_change_m"],
        "Physical occluder motion did not reveal background",
    )
    require(changed_camera.stamp.tick == control.runtime.tick, "Camera timestamp does not match scheduled capture")
    require(initial.cameras["inspection"].stamp.tick == 0, "Earlier packet mutated")
    if viewer is not None:
        viewer.update(changed)
    # A separate restricted view of the same physical scene must not expose localization or RGB.
    restricted = IsaacLabObservations(
        scene,
        ObservationConfig(joint_state=False, imu=False, cameras=(CameraPermission("inspection", rgb=False),)),
        dt,
        base=BaseLink(scene["robot"], robot_description.base_link),
        cameras=control.pipeline.cameras,  # The same render products, not a second set.
    )
    restricted.update(control.runtime.episode_id, control.runtime.tick)
    public = restricted.observe()
    require(public.localization is None and public.joints is None and public.imu is None, "Forbidden state leaked")
    require(
        public.cameras["inspection"].rgb is None and set(public.cameras) == {"inspection"},
        "Forbidden camera product leaked",
    )
    require(all("world" not in (edge.source, edge.target) for edge in public.transforms), "World transform leaked")
    try:
        public.transform("world", "inspection_optical", public.cameras["inspection"].stamp)
    except KeyError:
        pass
    else:
        raise AssertionError("Restricted packet resolved a privileged transform")
    torch.manual_seed(diagnostic["seed"])
    control.reset(settling_seconds=diagnostic["settling_seconds"])
    reset = control.observe()
    require(
        reset.stamp.episode_id == initial.stamp.episode_id + 1 and reset.stamp.tick == 0,
        "Reset retained episode history",
    )
    reset_depth = float(reset.cameras["inspection"].depth[y, x])
    require(abs(reset_depth - initial_depth) < diagnostic["plane_tolerance_m"], "Reset retained moved obstacle state")
    report.update(
        depth_plane_error_m=plane_error,
        initial_depth_m=initial_depth,
        revealed_depth_m=changed_depth,
        reset_depth_m=reset_depth,
        revealed=describe(changed),
        reset=describe(reset),
        permissions_checked=True,
        result=control.runtime.result(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("depth_plane_error_m", "initial_depth_m", "revealed_depth_m", "reset_depth_m")
            }
        ),
        flush=True,
    )
    return "passed"


def launch_obstacle(scene, diagnostic):
    velocity = torch.tensor(
        [[0, 0, diagnostic["lift_velocity_mps"], 0, 0, 0]], device=scene.device, dtype=torch.float32
    )
    scene["obstacle"].write_root_velocity_to_sim_index(root_velocity=velocity)


check.data["preset"] = args.preset
check.run(lambda: run(check.data))
