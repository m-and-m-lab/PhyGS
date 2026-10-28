# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trusted IsaacLab observation adapter; never passes simulator objects to agents."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch

from phygs.observations.localization import GroundTruthLocalization
from phygs.observations.types import (
    CameraObservation,
    ImuObservation,
    JointObservation,
    ObservationConfig,
    ObservationPacket,
    Stamp,
    Transform,
)
from phygs.runtime.timing import duration_ticks

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from phygs.simulation.base import BaseLink
    from phygs.simulation.cameras import RenderCameras
    from phygs.simulation.fabric import FabricTransformWrites


def pose_matrix(position: torch.Tensor, quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Build a homogeneous transform from translation [m] and quaternion (x,y,z,w)."""
    from isaaclab.utils.math import matrix_from_quat

    matrix = torch.eye(4, device=position.device, dtype=position.dtype)
    matrix[:3, :3] = matrix_from_quat(quaternion_xyzw)
    matrix[:3, 3] = position
    return matrix


class IsaacLabObservations:
    """Capture permitted products on runtime ticks; reads never advance physics.

    The current adapter supports one robot, pinhole cameras, and a body-aligned IMU. ``base`` reads the
    robot base (the ``body`` frame): its world pose and velocity (:class:`~.base.BaseLink`).
    Captured tensors are copied, and each public read returns a separate snapshot. Cameras are
    read from ``cameras``, by default :class:`~phygs.simulation.cameras.RenderCameras`
    for the permitted names, which renders at each capture. With ``fabric_writes``, PhysX writes
    body poses for rendering only in the steps before each camera capture; use it only when
    nothing else renders.
    """

    def __init__(
        self,
        scene: InteractiveScene,
        config: ObservationConfig,
        physics_dt: float,
        *,
        base: BaseLink,
        cameras: RenderCameras | None = None,
        fabric_writes: FabricTransformWrites | None = None,
    ) -> None:
        if scene.num_envs != 1:
            raise ValueError("Observations currently require exactly one environment.")
        self._fabric_writes = fabric_writes if config.cameras else None
        self._scene = scene
        self._device = scene.device
        self._robot = scene["robot"]
        self._base = base
        self._config = config
        self._dt = physics_dt
        self._state_ticks = duration_ticks(config.state_period, physics_dt)
        self._camera_ticks = duration_ticks(config.camera_period, physics_dt)
        self._localization = GroundTruthLocalization() if config.localization == "ground_truth" else None
        if config.imu and "observation_imu" not in scene.sensors:
            raise ValueError("IMU permission requires the observation_imu scene sensor.")
        names = tuple(permission.name for permission in config.cameras)
        if cameras is None and names:
            from phygs.simulation.cameras import RenderCameras

            cameras = RenderCameras(scene, names)
        self._cameras = cameras
        for permission in config.cameras:
            if permission.name not in (cameras.names if cameras is not None else ()):
                raise ValueError(f"Missing configured camera: {permission.name}")
            required = {"rgb"} if permission.rgb else set()
            if permission.depth:
                required.add("distance_to_image_plane")
            if not required.issubset(cameras.data_types(permission.name)):
                raise ValueError(f"Camera {permission.name} does not produce {sorted(required)}")
        self._packet: ObservationPacket | None = None
        self._camera_world: tuple[Transform, ...] = ()

    @property
    def cameras(self) -> RenderCameras | None:
        """The camera source; another adapter on the same scene may share it."""
        return self._cameras

    @property
    def configuration(self) -> dict:
        """Return serializable permissions, localization mode, and capture periods [s]."""
        return asdict(self._config)

    def reset(self, episode_id: int) -> None:
        """Discard all cached measurements and capture the settled episode at time zero."""
        self._packet = None
        self._camera_world = ()
        self.update(episode_id, 0)

    def update(self, episode_id: int, tick: int) -> None:
        """Capture due sensors at the supplied runtime tick without stepping physics."""
        if self._packet is not None and self._packet.stamp.episode_id != episode_id:
            raise RuntimeError("Reset observations before starting another episode.")
        stamp = Stamp(episode_id, tick, tick * self._dt)
        previous = self._packet
        joints = previous.joints if previous else None
        imu = previous.imu if previous else None
        localization = previous.localization if previous else None
        cameras = previous.cameras if previous else {}
        state_due = tick % self._state_ticks == 0
        camera_due = bool(self._config.cameras) and tick % self._camera_ticks == 0
        body_pose = None
        if (state_due and self._localization is not None) or camera_due:
            pose = self._base.pose()
            body_pose = pose_matrix(pose[:3], pose[3:])
        if state_due:
            if self._config.joint_state:
                positions = self._robot.data.joint_pos.torch[0].clone()
                velocities = self._robot.data.joint_vel.torch[0].clone()
                joints = JointObservation(
                    stamp,
                    tuple(self._robot.joint_names),
                    positions,
                    velocities,
                    bool(torch.isfinite(positions).all() and torch.isfinite(velocities).all()),
                )
            if self._config.imu:
                data = self._scene["observation_imu"].data
                gyro, acceleration = data.ang_vel_b.torch[0].clone(), data.lin_acc_b.torch[0].clone()
                imu = ImuObservation(
                    stamp, gyro, acceleration, bool(torch.isfinite(gyro).all() and torch.isfinite(acceleration).all())
                )
            if self._localization is not None:
                localization = self._localization.observe(stamp, body_pose, self._base.linear_velocity_b())
        if camera_due:
            self._cameras.render()
            cameras = {}
            self._camera_world = (
                (Transform("world", "body", stamp, body_pose.clone(), "ground_truth_localization"),)
                if self._localization
                else ()
            )
            for permission in self._config.cameras:
                image = self._cameras.read(permission.name)
                frame = f"{permission.name}_optical"
                extrinsics = Transform(
                    "body",
                    frame,
                    stamp,
                    torch.linalg.solve(body_pose, image.world_from_camera),
                    "ideal_robot_kinematics",
                )
                rgb = image.rgb if permission.rgb else None
                depth, mask = image.depth if permission.depth else None, None
                if depth is not None:  # Too near or no return: no depth (0), as the sensor reports it.
                    # On the renderer's device, then one copy each to the scene device: masking a CPU
                    # copy cost about 1.5 ms per 640x480 frame, on the GPU about 0.3 ms with the copy.
                    mask = torch.isfinite(depth) & (depth > 0) & (depth >= self._config.depth_min)
                    depth = torch.where(mask, depth, torch.zeros_like(depth)).to(self._device)
                    mask = mask.to(self._device)
                intrinsics = image.intrinsics
                valid = bool(torch.isfinite(intrinsics).all() and torch.isfinite(extrinsics.matrix).all())
                if not valid or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                    raise RuntimeError(f"Invalid calibration for {permission.name}")
                cameras[permission.name] = CameraObservation(
                    stamp, frame, rgb, depth, mask, intrinsics, extrinsics, valid
                )
        transforms = tuple(camera.extrinsics for camera in cameras.values()) + self._camera_world
        if localization is not None:
            # State and image rates may differ; preserve localization at both capture times.
            if not self._camera_world or localization.stamp != self._camera_world[0].stamp:
                transforms += (localization.pose,)
        self._packet = ObservationPacket(stamp, joints, imu, localization, cameras, transforms)
        if self._fabric_writes is not None:  # Rendering reads only the poses of the steps before a capture.
            self._fabric_writes.set(self._fabric_writes.before_capture(tick, self._camera_ticks))

    def observe_internal(self) -> ObservationPacket:
        """Return the trusted packet without copying held camera tensors."""
        if self._packet is None:
            raise RuntimeError("Reset the environment before requesting observations.")
        return self._packet

    def observe(self) -> ObservationPacket:
        """Return a detached permitted snapshot; fail before the first episode reset."""
        return copy.deepcopy(self.observe_internal())
