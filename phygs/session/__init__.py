# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What an agent gets: the session, its target types and its errors. Agents import from here only.

``run(session)`` receives an :class:`AgentSession`; everything else here is a type an agent builds
(targets, trajectories) or catches (:class:`EpisodeEnded`). No Isaac Sim imports; the agent runs in
its own thread and every call is carried out on the main thread by the world (:mod:`.session`).
"""

from phygs.session.session import (
    AgentFailed,
    AgentSession,
    ArticulatedPart,
    Command,
    EndEffectorTrajectory,
    EpisodeEnded,
    HeldObject,
    JointTrajectory,
    ObjectTarget,
    PixelTarget,
    PointTarget,
    PublicTask,
    RobotInfo,
    SkillFailure,
    TargetHint,
    TrajectoryRefused,
    entry_depth,
    motion_failure,
    unit_approach,
)

__all__ = [
    "AgentFailed",
    "AgentSession",
    "ArticulatedPart",
    "Command",
    "EndEffectorTrajectory",
    "EpisodeEnded",
    "HeldObject",
    "JointTrajectory",
    "ObjectTarget",
    "PixelTarget",
    "PointTarget",
    "PublicTask",
    "RobotInfo",
    "SkillFailure",
    "TargetHint",
    "TrajectoryRefused",
    "entry_depth",
    "motion_failure",
    "unit_approach",
]
