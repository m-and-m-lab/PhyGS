# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Isaac Lab scene and simulation for a robot description: the robot, its cameras and IMU, a
ground plane and a dome light."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

from phygs.simulation.cameras import RenderCameraCfg

if TYPE_CHECKING:
    from phygs.robots.description import RobotDescription

ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"
DATA_TYPES = ("rgb", "distance_to_image_plane")


def render_camera_cfg(camera: dict) -> RenderCameraCfg:
    """Build a render camera from its configuration; ``offset`` places a spawned ``pinhole`` camera.

    The offset rotation (x, y, z, w) is in its ``convention`` (``ros`` by default, ``opengl`` or
    ``world``) relative to the parent prim, as IsaacLab's camera sensor applied it.
    """
    import torch
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    pinhole = None
    offset = {"pos": (0.0, 0.0, 0.0), "rot": (0.0, 0.0, 0.0, 1.0), "convention": "ros", **camera.get("offset", {})}
    rotation = convert_camera_frame_orientation_convention(
        torch.tensor([offset["rot"]], dtype=torch.float32), origin=offset["convention"], target="opengl"
    )[0]
    if "pinhole" in camera:
        parameters = dict(camera["pinhole"])
        parameters["clipping_range"] = tuple(parameters["clipping_range"])
        pinhole = sim_utils.PinholeCameraCfg(**parameters)
        if pinhole.vertical_aperture is None:
            pinhole.vertical_aperture = pinhole.horizontal_aperture * camera["height"] / camera["width"]
    return RenderCameraCfg(
        prim_path=camera["prim_path"],
        spawn=pinhole,
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(offset["pos"]), rot=tuple(rotation.tolist())),
        width=camera["width"],
        height=camera["height"],
        data_types=list(camera["data_types"]),
    )


def robot_cameras(robot: RobotDescription, names: tuple[str, ...], data_types=DATA_TYPES) -> dict[str, dict]:
    """Render camera configurations (:func:`render_camera_cfg`) of the robot's cameras ``names``."""
    unknown = set(names) - set(robot.cameras)
    if unknown:
        raise ValueError(f"Unknown cameras {sorted(unknown)}; {robot.name} has {sorted(robot.cameras)}.")
    cameras = {}
    for name in names:
        camera = robot.cameras[name]
        config = dict(prim_path=f"{ROBOT_PRIM}/{camera.prim}", width=camera.width, height=camera.height)
        if camera.pinhole is not None:
            config.update(pinhole=dict(camera.pinhole), offset=dict(camera.offset or {}))
        cameras[name] = dict(config, data_types=list(data_types))
    return cameras


def make_scene_cfg(
    robot: RobotDescription,
    *,
    physics_dt: float,
    light: dict,
    position: tuple[float, float, float] | None = None,
    heading: float = 0.0,
    cameras: dict[str, dict] | None = None,
    imu: bool = False,
) -> type[InteractiveSceneCfg]:
    """The scene class: a ground plane, a dome light (``light``: intensity, color), the robot at
    ``position`` [m] (default: the origin at its spawn height) facing ``heading`` [rad], its arm
    stowed, with its actuators at the physics timestep ``physics_dt`` [s], and the render ``cameras``
    (name -> :func:`render_camera_cfg` configuration; :func:`robot_cameras` builds the robot's own).
    ``imu`` adds the IMU sensor ``observation_imu`` on the robot's IMU link."""
    position = (0.0, 0.0, robot.spawn_height) if position is None else tuple(map(float, position))
    articulation = ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot.usd),
            activate_contact_sensors=robot.articulation.contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=robot.articulation.self_collisions,
                solver_position_iteration_count=robot.articulation.solver_position_iterations,
                solver_velocity_iteration_count=robot.articulation.solver_velocity_iterations,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=(0.0, 0.0, math.sin(heading / 2), math.cos(heading / 2)),
            joint_pos={**robot.initial_joints, **dict(robot.arm.transit_posture)},
            joint_vel={".*": 0.0},
        ),
        actuators=robot.plugin(robot.actuators)(physics_dt),
    )

    dome = sim_utils.DomeLightCfg(intensity=float(light["intensity"]), color=tuple(map(float, light["color"])))

    @configclass
    class RobotSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=dome)
        robot: ArticulationCfg = articulation

    sensors = {}
    if imu:
        sensors["observation_imu"] = ImuCfg(prim_path=f"{ROBOT_PRIM}/{robot.imu_link}", update_period=0.0)
    for name, camera in (cameras or {}).items():
        if hasattr(RobotSceneCfg, name) or name in {"body", "world", "observation_imu"}:
            raise ValueError(f"Camera name conflicts with a scene entity or reserved frame: {name}")
        sensors[name] = render_camera_cfg(camera)
    return configclass(type("RobotSensorSceneCfg", (RobotSceneCfg,), sensors))


def simulation_cfg(dt: float, **kwargs) -> sim_utils.SimulationCfg:
    """The simulation at timestep ``dt`` [s]; ``kwargs`` as :class:`~isaaclab.sim.SimulationCfg`.

    PhysX applies gravity in every TGS position iteration: applied once per step, a step's gravity
    stayed in the reported velocities of Spot standing on PhysX leg drives (0.04 rad/s at arm
    joints and 0.01 rad/s body pitch while still; 2026-10-22).
    """
    from isaaclab_physx.physics import PhysxCfg

    return sim_utils.SimulationCfg(dt=dt, physics=PhysxCfg(enable_external_forces_every_iteration=True), **kwargs)
