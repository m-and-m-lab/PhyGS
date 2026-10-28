# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent-visible measurements; all transforms map source coordinates into target."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import torch


@dataclass(frozen=True)
class CameraPermission:
    """Allow specific image products from one configured camera."""

    name: str
    rgb: bool = True
    depth: bool = True

    def __post_init__(self) -> None:
        if not self.name or not (self.rgb or self.depth):
            raise ValueError("A camera permission needs a name and at least one image product.")


@dataclass(frozen=True)
class ObservationConfig:
    """Permitted products and capture periods [s], selected by trusted task setup.

    ``depth_min`` [m]: depth nearer than this is no depth (0, invalid), as the robot's Orbbec Gemini 2
    cameras return none there (NVIDIA lists their ideal range as 0.15-10 m); every consumer sees only
    this, never the rendered depth. The renderer draws to 5 mm, so images show near surfaces.
    """

    joint_state: bool = True
    imu: bool = True
    localization: Literal["none", "ground_truth"] = "none"
    cameras: tuple[CameraPermission, ...] = ()
    state_period: float = 0.02
    camera_period: float = 0.032
    depth_min: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if not all(math.isfinite(value) and value > 0 for value in (self.state_period, self.camera_period)):
            raise ValueError("Observation periods must be positive and finite.")
        if not math.isfinite(self.depth_min) or self.depth_min < 0:
            raise ValueError("The minimum depth must be finite and nonnegative.")
        if self.localization not in {"none", "ground_truth"}:
            raise ValueError(f"Unsupported localization configuration: {self.localization}")
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError("Camera permissions must have unique names.")


@dataclass(frozen=True)
class ObservationAccess:
    """Products available to an agent or service, independently of capture rates."""

    joint_state: bool = False
    imu: bool = False
    localization: Literal["none", "ground_truth", "estimated"] = "none"
    cameras: tuple[CameraPermission, ...] = ()

    def __post_init__(self) -> None:
        if self.localization not in {"none", "ground_truth", "estimated"}:
            raise ValueError("Unsupported localization access mode.")
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if len({item.name for item in self.cameras}) != len(self.cameras):
            raise ValueError("Camera access names must be unique.")

    @classmethod
    def from_config(cls, config: ObservationConfig) -> ObservationAccess:
        """Select all products in a capture configuration."""
        return cls(config.joint_state, config.imu, config.localization, config.cameras)

    def require(self, available: ObservationAccess, *, consumer: str) -> None:
        """Reject requested products outside the available information policy."""
        if self.joint_state and not available.joint_state or self.imu and not available.imu:
            raise PermissionError(f"{consumer} requires unavailable robot observations.")
        if self.localization != "none" and self.localization != available.localization:
            raise PermissionError(f"{consumer} requires unavailable {self.localization} localization.")
        cameras = {item.name: item for item in available.cameras}
        for item in self.cameras:
            permitted = cameras.get(item.name)
            if permitted is None or item.rgb and not permitted.rgb or item.depth and not permitted.depth:
                raise PermissionError(f"{consumer} requires unavailable camera products: {item.name}")

    def select(self, packet: ObservationPacket) -> ObservationPacket:
        """Select a detached view, retaining only permitted capture-time transforms.

        Sensor extrinsics remain body-relative. Localization transforms are
        included only for the explicitly selected provider; unrelated transform
        graph edges are never copied into an agent view.
        """
        import copy

        if self.joint_state and packet.joints is None or self.imu and packet.imu is None:
            raise RuntimeError("Required robot observations are missing.")
        cameras = {}
        for item in self.cameras:
            camera = packet.cameras[item.name]
            if item.rgb and camera.rgb is None or item.depth and camera.depth is None:
                raise PermissionError(f"Missing permitted camera product: {item.name}")
            if camera.extrinsics.target != "body" or camera.extrinsics.source != camera.frame:
                raise ValueError("Sensor extrinsics must be body-relative.")
            cameras[item.name] = replace(
                camera,
                rgb=camera.rgb if item.rgb else None,
                depth=camera.depth if item.depth else None,
                depth_valid=camera.depth_valid if item.depth else None,
            )
        localization = packet.localization if self.localization != "none" else None
        if self.localization != "none" and (localization is None or localization.mode != self.localization):
            raise RuntimeError("Selected localization provider did not produce the requested mode.")
        transforms = tuple(camera.extrinsics for camera in cameras.values())
        if localization is not None:
            times = {localization.stamp, *(camera.stamp for camera in cameras.values())}
            transforms += tuple(
                edge
                for edge in packet.transforms
                if edge.target == localization.pose.target
                and edge.source == "body"
                and edge.information_source == localization.pose.information_source
                and edge.stamp in times
            )
        return copy.deepcopy(
            ObservationPacket(
                packet.stamp,
                packet.joints if self.joint_state else None,
                packet.imu if self.imu else None,
                localization,
                cameras,
                transforms,
            )
        )


@dataclass(frozen=True)
class Stamp:
    """Capture time [s] within an episode, excluding reset settling."""

    episode_id: int
    tick: int
    time_seconds: float


@dataclass(frozen=True)
class Transform:
    """Homogeneous target-from-source matrix; translation [m], rotation unitless."""

    target: str
    source: str
    stamp: Stamp
    matrix: torch.Tensor
    information_source: str


@dataclass(frozen=True)
class JointObservation:
    """Ordered revolute joint positions [rad] and velocities [rad/s]."""

    stamp: Stamp
    names: tuple[str, ...]
    positions: torch.Tensor
    velocities: torch.Tensor
    valid: bool
    information_source: str = "simulated_joint_encoders"


@dataclass(frozen=True)
class ImuObservation:
    """Body-aligned gyro [rad/s] and proper acceleration [m/s^2]; +g upward at rest."""

    stamp: Stamp
    angular_velocity: torch.Tensor
    proper_acceleration: torch.Tensor
    valid: bool
    frame: str = "body"
    information_source: str = "ideal_simulated_imu"


@dataclass(frozen=True)
class LocalizationObservation:
    """Body pose in the provider reference frame and body linear velocity [m/s]."""

    stamp: Stamp
    pose: Transform
    linear_velocity: torch.Tensor
    valid: bool
    mode: str


@dataclass(frozen=True)
class CameraObservation:
    """RGB uint8 [H,W,3], optical-axis depth [m, H,W], calibration, and pixel validity.

    Optical frame is +X right, +Y down, +Z forward. Intrinsics use pixel coordinates.
    Missing products are None, never synthetic images. Extrinsics map optical to body.
    """

    stamp: Stamp
    frame: str
    rgb: torch.Tensor | None
    depth: torch.Tensor | None
    depth_valid: torch.Tensor | None
    intrinsics: torch.Tensor
    extrinsics: Transform
    valid: bool
    information_source: str = "simulated_rgbd_camera"

    def points(self, stride: int = 1, *, rows: int | None = None) -> torch.Tensor:
        """Unproject valid depth pixels into optical-frame points [m, N,3], row by row; with ``stride``
        only every stride-th row and column (``stride``² fewer points), with ``rows`` only the image's
        top ``rows`` rows."""
        if self.depth is None or self.depth_valid is None:
            raise PermissionError("Depth was not permitted for this camera.")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("Pixel stride must be a positive integer.")
        valid = self.depth_valid if rows is None else self.depth_valid[:rows]
        rows, cols = torch.where(valid[::stride, ::stride])
        rows, cols = rows * stride, cols * stride
        pixels = torch.stack((cols, rows, torch.ones_like(rows)), dim=-1).to(self.depth.dtype)
        rays = torch.linalg.solve(self.intrinsics, pixels.T).T
        return rays * self.depth[rows, cols, None]


@dataclass(frozen=True)
class ObservationPacket:
    """Latest permitted samples, each retaining its capture time rather than polling time."""

    stamp: Stamp
    joints: JointObservation | None
    imu: ImuObservation | None
    localization: LocalizationObservation | None
    cameras: dict[str, CameraObservation]
    transforms: tuple[Transform, ...]

    def transform(self, target: str, source: str, stamp: Stamp) -> torch.Tensor:
        """Return target-from-source at an exact capture time; translation [m].

        Unknown frames, unavailable times, and cross-episode queries fail explicitly.
        No interpolation or privileged world-frame access is implied.
        """
        edges = [edge for edge in self.transforms if edge.stamp == stamp]
        if stamp.episode_id != self.stamp.episode_id:
            raise KeyError("Transform belongs to another episode.")
        frames = {frame for edge in edges for frame in (edge.target, edge.source)}
        if source not in frames or target not in frames:
            raise KeyError(f"Unavailable frames at {stamp}: {source} -> {target}")
        identity = torch.eye(4, dtype=edges[0].matrix.dtype, device=edges[0].matrix.device)
        queue = deque([(source, identity)])
        visited = {source}
        while queue:
            frame, matrix = queue.popleft()
            if frame == target:
                return matrix
            for edge in edges:
                if edge.source == frame:
                    neighbor, step = edge.target, edge.matrix
                elif edge.target == frame:
                    neighbor, step = edge.source, torch.linalg.inv(edge.matrix)
                else:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, step @ matrix))
        raise KeyError(f"No permitted transform path: {source} -> {target}")


class ObservationReader(Protocol):
    """Read permitted observations without controlling the environment clock."""

    def observe(self) -> ObservationPacket: ...
