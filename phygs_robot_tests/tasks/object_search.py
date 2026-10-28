# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-search task and independently grounded image-evidence evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from phygs.observations.types import ObservationPacket, Stamp


@dataclass(frozen=True)
class ObjectSearchTask:
    """Public target description, camera, budget [s], surface tolerance [m], and actions."""

    scene_id: str
    initial_state_id: str
    target_description: str
    camera: str
    time_budget: float = 45.0
    allowed_actions: tuple[str, ...] = ("navigation", "motion")
    min_support_pixels: int = 3
    surface_tolerance: float = 0.08

    def __post_init__(self) -> None:
        if not all((self.scene_id, self.initial_state_id, self.target_description, self.camera)):
            raise ValueError("Object search requires scene, initial state, target description, and camera.")
        if not self.allowed_actions or not set(self.allowed_actions) <= {"navigation", "motion"}:
            raise ValueError("Unsupported object-search action interfaces.")
        if not math.isfinite(self.time_budget) or self.time_budget <= 0:
            raise ValueError("Object-search time budget must be positive and finite.")
        if type(self.min_support_pixels) is not int or not 1 <= self.min_support_pixels <= 9:
            raise ValueError("Support count must be between one and nine pixels.")
        if not math.isfinite(self.surface_tolerance) or self.surface_tolerance < 0:
            raise ValueError("Surface tolerance must be nonnegative and finite.")


@dataclass(frozen=True)
class TargetBounds:
    """Trusted world-frame axis-aligned target bounds [m]. Never agent-visible."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.minimum) != 3 or len(self.maximum) != 3:
            raise ValueError("Target bounds require three coordinates per corner.")
        if any(not math.isfinite(a) or not math.isfinite(b) or a >= b for a, b in zip(self.minimum, self.maximum)):
            raise ValueError("Target bounds must be finite and have positive size.")


@dataclass(frozen=True)
class TargetDeclaration:
    """Agent-submitted image pixel and capture stamp; no world pose is accepted."""

    camera: str
    stamp: Stamp
    pixel_x: int
    pixel_y: int

    def __post_init__(self) -> None:
        if not self.camera or type(self.pixel_x) is not int or type(self.pixel_y) is not int:
            raise ValueError("Target evidence needs a camera and integer pixel coordinates.")


@dataclass(frozen=True)
class _EvidenceFrame:
    stamp: Stamp
    depth: torch.Tensor
    valid: torch.Tensor
    intrinsics: torch.Tensor
    world_from_camera: torch.Tensor
    bounds: TargetBounds


class ObjectSearchEvaluator:
    """Validate declarations against captured depth and private physical target bounds.

    The evaluator records a trusted observation view at capture time. A declaration
    succeeds only if its exact stamped pixel and nearby support rays terminate on
    the target's physical bounds. Occluders therefore cannot count as target evidence.
    """

    def __init__(self, task: ObjectSearchTask, *, episode_id: int, history_frames: int = 64) -> None:
        if type(episode_id) is not int or episode_id < 1:
            raise ValueError("Object search requires a positive episode ID.")
        if type(history_frames) is not int or history_frames < 1:
            raise ValueError("Evidence history must retain at least one frame.")
        self.task = task
        self.episode_id = episode_id
        self.history_frames = history_frames
        self.status = "running"
        self.reason = ""
        self.elapsed_seconds = 0.0
        self.declaration: TargetDeclaration | None = None
        self._frames: dict[Stamp, _EvidenceFrame] = {}

    def record(self, packet: ObservationPacket, bounds: TargetBounds) -> bool:
        """Record one trusted camera capture and concurrent target geometry [m]."""
        camera = packet.cameras[self.task.camera]
        stamp = camera.stamp
        if stamp.episode_id != self.episode_id or packet.stamp.episode_id != self.episode_id:
            raise ValueError("Object-search observation belongs to another episode.")
        if stamp in self._frames:
            return False
        if self._frames and stamp.time_seconds <= next(reversed(self._frames)).time_seconds:
            raise ValueError("Object-search camera timestamps must increase.")
        if not camera.valid or camera.rgb is None or camera.depth is None or camera.depth_valid is None:
            raise ValueError("Object search requires a valid RGB-D camera capture.")
        if camera.depth.shape != camera.depth_valid.shape or camera.rgb.shape[:2] != camera.depth.shape:
            raise ValueError("Object-search camera products have inconsistent image dimensions.")
        transform = packet.transform("world", camera.frame, stamp)
        if transform.shape != (4, 4) or not bool(torch.isfinite(transform).all()):
            raise ValueError("Object-search camera pose must be a finite world transform.")
        frame = _EvidenceFrame(
            stamp,
            camera.depth.detach().clone(),
            camera.depth_valid.detach().clone(),
            camera.intrinsics.detach().clone(),
            transform.detach().clone(),
            bounds,
        )
        self._frames[stamp] = frame
        if len(self._frames) > self.history_frames:
            del self._frames[next(iter(self._frames))]
        return True

    def update(self, time_seconds: float) -> str:
        """Advance the evaluator clock [s] without reading navigation feedback."""
        if not math.isfinite(time_seconds) or time_seconds < self.elapsed_seconds:
            raise ValueError("Object-search time must increase and remain finite.")
        self.elapsed_seconds = time_seconds
        if self.status == "running" and time_seconds >= self.task.time_budget:
            self.status, self.reason = "timed_out", "Search budget exhausted"
        return self.status

    @staticmethod
    def _hits_target(frame: _EvidenceFrame, x: int, y: int, tolerance: float) -> bool:
        if not bool(frame.valid[y, x]):
            return False
        depth = frame.depth[y, x]
        if not bool(torch.isfinite(depth)) or float(depth) <= 0:
            return False
        pixel = torch.tensor([x, y, 1], dtype=depth.dtype, device=depth.device)
        optical = torch.linalg.solve(frame.intrinsics, pixel) * depth
        world = frame.world_from_camera[:3, :3] @ optical + frame.world_from_camera[:3, 3]
        return all(
            a - tolerance <= float(value) <= b + tolerance
            for value, a, b in zip(world, frame.bounds.minimum, frame.bounds.maximum)
        )

    def submit(self, declaration: TargetDeclaration) -> str:
        """Accept or reject stamped image evidence; a wrong target ends the search."""
        if self.status != "running":
            raise RuntimeError("Object-search task is already terminal.")
        if declaration.camera != self.task.camera or declaration.stamp.episode_id != self.episode_id:
            raise ValueError("Target declaration uses an unavailable camera or episode.")
        frame = self._frames.get(declaration.stamp)
        if frame is None:
            raise ValueError("Target declaration references an uncaptured or expired frame.")
        height, width = frame.depth.shape
        x, y = declaration.pixel_x, declaration.pixel_y
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("Target declaration pixel is outside the captured image.")
        self.declaration = declaration
        support = sum(
            self._hits_target(frame, col, row, self.task.surface_tolerance)
            for row in range(max(0, y - 1), min(height, y + 2))
            for col in range(max(0, x - 1), min(width, x + 2))
        )
        if self._hits_target(frame, x, y, self.task.surface_tolerance) and support >= self.task.min_support_pixels:
            self.status, self.reason = "succeeded", "Stamped depth evidence matches the physical target"
        else:
            self.status, self.reason = "false_positive", "Declared image evidence does not reach the physical target"
        return self.status

    def result(self) -> dict:
        """Return task identity, outcome, and evidence provenance."""
        return {
            "task": asdict(self.task),
            "episode_id": self.episode_id,
            "status": self.status,
            "reason": self.reason,
            "elapsed_seconds": self.elapsed_seconds,
            "declaration": asdict(self.declaration) if self.declaration else None,
            "frames_recorded": len(self._frames),
        }
