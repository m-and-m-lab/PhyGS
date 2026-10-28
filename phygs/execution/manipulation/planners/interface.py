# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Common contract for the planning libraries (MoveIt, cuRobo); standard library only.

Each backend maps these calls onto its library. Planning runs in a separate process,
so requests and results cross a pipe as JSON; poses are [x, y, z, qx, qy, qz, qw] [m]
in the robot body frame, joint values in [rad], times in [s].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Protocol, TypedDict


@dataclass(frozen=True)
class DepthImage:
    """One depth image for the planner's depth map.

    ``path``: ``.npy`` file, float32 [H, W], depth along the optical axis [m]; 0 marks no data.
    Pixels on the robot or its held object may stay: the planner removes them (see :class:`PlanRequest`).
    ``intrinsics``: 3x3 pinhole matrix [px].
    ``pose``: optical frame (x right, y down, z forward) in the body frame.
    """

    path: str
    intrinsics: list[list[float]]
    pose: list[float]


@dataclass(frozen=True)
class HeldShape:
    """Held object for collision checking only, in the wrist frame.

    ``box``: size [m] and pose (MoveIt attaches it).
    ``spheres``: [x, y, z, r] [m] covering the box (cuRobo).
    The planner keeps its clearance between the object and obstacles, as for the robot.
    """

    box: dict
    spheres: list[list[float]]


@dataclass(frozen=True)
class PlanRequest:
    """Plan from the measured start to a joint goal or a wrist pose goal.

    The plan avoids the boxes and a depth map built for this request only: the planner clears its
    map and fuses ``images`` (none: an empty map). The images are taken at ``start`` with the arm
    still; the library's self filter drops their pixels on the robot and on the held object
    (MoveIt: the Octomap shape mask on the link frames; cuRobo: ``RobotSegmenter`` on the
    collision spheres), so the map holds the scene only. Unobserved space is free.
    """

    kind: ClassVar[str] = "plan"

    start: list[float]
    boxes: list[dict]
    goal: list[float] | None = None
    goal_pose: list[float] | None = None
    gripper: float | None = None  # Measured gripper angle [rad]; None uses the model's angle.
    held: HeldShape | None = None
    # Measured angles [rad] of the model's other locked joints (the legs); None keeps the last ones.
    locked_joints: dict[str, float] | None = None
    images: list[DepthImage] | None = None
    # Where the planner saves the obstacle voxels this plan avoids, for display: body-frame centres
    # [m], float32 [N,3] .npy. None: not saved. cuRobo saves them; MoveIt does not.
    world_path: str | None = None

    def __post_init__(self) -> None:
        if (self.goal is None) == (self.goal_pose is None):
            raise ValueError("A plan request needs exactly one of goal and goal_pose.")

    def to_json(self) -> dict:
        return {"kind": self.kind, **asdict(self)}

    @classmethod
    def from_json(cls, data: dict) -> PlanRequest:
        held, images = data.get("held"), data.get("images")
        return cls(
            **{
                **data,
                "held": None if held is None else HeldShape(**held),
                "images": None if images is None else [DepthImage(**image) for image in images],
            }
        )


@dataclass(frozen=True)
class IkRequest:
    """Arm joints within joint limits that put the wrist at a pose, solved from the seed joints."""

    kind: ClassVar[str] = "ik"

    pose: list[float]
    seed: list[float]

    def to_json(self) -> dict:
        return {"kind": self.kind, **asdict(self)}

    @classmethod
    def from_json(cls, data: dict) -> IkRequest:
        return cls(**data)


Request = PlanRequest | IkRequest


def request_from_json(data: dict) -> Request:
    """Rebuild a request sent through the pipe; its ``kind`` names the planner method to call."""
    data = dict(data)
    kind = data.pop("kind")
    requests = {request.kind: request for request in (PlanRequest, IkRequest)}
    if kind not in requests:
        raise ValueError(f"Unknown planner request kind {kind!r}.")
    return requests[kind].from_json(data)


class PlanResult(TypedDict, total=False):
    """Plan outcome; on success, timed joint samples of the arm joints."""

    success: bool
    reason: str  # Why planning failed, e.g. "no_plan_within_budget" or "start_collision".
    plan_seconds: float  # Including the depth map.
    map_seconds: float  # Clearing the depth map and fusing the request's images.
    joint_names: list[str]
    times: list[float]
    positions: list[list[float]]


class IkResult(TypedDict, total=False):
    """IK outcome; on success, the arm joint values."""

    success: bool
    joints: list[float]
    seconds: float


class ManipulationPlanner(Protocol):
    """A planning library behind the worker; built once from the prepared robot model."""

    def plan(self, request: PlanRequest) -> PlanResult: ...

    def ik(self, request: IkRequest) -> IkResult: ...

    def close(self) -> None: ...
