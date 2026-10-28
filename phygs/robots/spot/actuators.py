# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot's actuator groups (``robot.yaml`` ``actuators``): the twelve legs as one IsaacLab actuator
group, and the arm and gripper drives."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
    IdealPDActuator,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
    RemotizedPDActuatorCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions
from isaaclab_assets.robots.spot import joint_parameter_lookup


class SpotLegActuator(IdealPDActuator):
    """Spot's hip and knee actuators in one group, equal to IsaacLab's two groups.

    Hips behave as :class:`~isaaclab.actuators.DelayedPDActuator`: delayed position, velocity
    and effort set-points, PD torque, clipped to :attr:`effort_limit`. Knees behave as
    :class:`~isaaclab.actuators.RemotizedPDActuator`: the same delayed PD, clamped to the
    joint-angle-dependent limit of :attr:`SpotLegActuatorCfg.joint_parameter_lookup`. Hips and
    knees draw separate delays at reset, as the two IsaacLab groups do, in the same order.

    IsaacLab's articulation spends most of an explicit actuator's cost per group (index gathers
    and kernel launches), so one group for all twelve legs halves it. The delay history is a
    ring buffer that follows the indexing rules of :class:`~isaaclab.utils.buffers.DelayBuffer`:
    the first set-point after a reset fills the history, and a delay longer than the set-points
    since the reset returns the oldest one.

    Joints handed to the PhysX joint drives (:meth:`set_physx_drive`) get no effort from this
    actuator: their set-point goes to PhysX undelayed, with zero desired velocity, while the delay
    history keeps running. The caller writes the PhysX drive gains.
    """

    cfg: SpotLegActuatorCfg

    def __init__(self, cfg: SpotLegActuatorCfg, joint_names: list[str], *args, **kwargs):
        super().__init__(cfg, joint_names, *args, **kwargs)
        from isaaclab.utils import LinearInterpolation

        knee = re.compile(cfg.knee_joint_expr)
        self._knee = torch.tensor([bool(knee.fullmatch(name)) for name in joint_names], device=self._device)
        if not self._knee.any() or self._knee.all():
            raise ValueError("SpotLegActuator needs hip and knee joints.")
        self._knee_cols = self._knee.nonzero().flatten()
        # As RemotizedPDActuator: knees have no velocity box; hips keep the joint's value. A copy:
        # without a configured limit, IsaacLab shares this tensor with the simulation limit.
        self.velocity_limit = self.velocity_limit.clone()
        self.velocity_limit[:, self._knee_cols] = torch.inf
        lookup = torch.tensor(cfg.joint_parameter_lookup, device=self._device)
        self._knee_torque_limit = LinearInterpolation(lookup[:, 0], lookup[:, 2], device=self._device)
        self._history = torch.zeros((3, cfg.max_delay + 1, self._num_envs, self.num_joints), device=self._device)
        self._head = 0
        self._pushes = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._lags = torch.zeros((self._num_envs, self.num_joints), dtype=torch.long, device=self._device)
        self._envs = torch.arange(self._num_envs, device=self._device).unsqueeze(1)
        self._joints = torch.arange(self.num_joints, device=self._device).unsqueeze(0)
        self._physx_drive = torch.zeros(self.num_joints, dtype=torch.bool, device=self._device)

    @property
    def time_lags(self) -> torch.Tensor:
        """Current delay per environment and joint [physics steps], shape [num_envs, num_joints]."""
        return self._lags

    def set_time_lags(self, hip: torch.Tensor | int, knee: torch.Tensor | int) -> None:
        """Set the hip and knee delays [physics steps], per environment or for all."""
        hip, knee = (torch.as_tensor(value, device=self._device).long().expand(self._num_envs) for value in (hip, knee))
        self._lags[:] = torch.where(self._knee, knee.unsqueeze(1), hip.unsqueeze(1))

    @property
    def physx_drive(self) -> torch.Tensor:
        """Joints driven by the PhysX joint drives instead of this actuator, shape [num_joints] (a copy)."""
        return self._physx_drive.clone()

    def set_physx_drive(self, joints: Sequence[int], enabled: bool) -> None:
        """Hand joints (columns of this group) to the PhysX joint drives, or back to this actuator."""
        self._physx_drive[list(joints)] = enabled

    def reset(self, env_ids: Sequence[int] | None):
        ids = slice(None) if env_ids is None or env_ids == slice(None) else env_ids
        count = self._num_envs if isinstance(ids, slice) else len(ids)
        low, high = self.cfg.min_delay, self.cfg.max_delay + 1
        hip = torch.randint(low, high, (count,), dtype=torch.int, device=self._device)
        knee = torch.randint(low, high, (count,), dtype=torch.int, device=self._device)
        self._lags[ids] = torch.where(self._knee, knee.long().unsqueeze(1), hip.long().unsqueeze(1))
        self._pushes[ids] = 0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        set_points = torch.stack(
            (control_action.joint_positions, control_action.joint_velocities, control_action.joint_efforts)
        )
        first = self._pushes == 0
        if torch.any(first):
            self._history[:, :, first] = set_points[:, first].unsqueeze(1)
        length = self._history.shape[1]
        self._head = (self._head + 1) % length
        self._history[:, self._head] = set_points
        self._pushes += 1
        lag = torch.clamp(torch.minimum(self._lags, (self._pushes - 1).unsqueeze(1)), min=0)
        delayed = self._history[:, (self._head - lag) % length, self._envs, self._joints]
        error_pos = delayed[0] - joint_pos
        error_vel = delayed[1] - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + delayed[2]
        effort = self._clip_effort(self.computed_effort)
        limit = self._knee_torque_limit.compute(joint_pos[:, self._knee_cols])
        effort[:, self._knee_cols] = torch.clamp(effort[:, self._knee_cols], min=-limit, max=limit)
        control_action.joint_positions = None
        control_action.joint_velocities = None
        if self._physx_drive.any():
            # PhysX drive gains are zero on the other joints, so their position target has no effect.
            self.computed_effort = torch.where(self._physx_drive, 0.0, self.computed_effort)
            effort = torch.where(self._physx_drive, 0.0, effort)
            control_action.joint_positions = set_points[0]
            control_action.joint_velocities = torch.zeros_like(set_points[1])
        self.applied_effort = effort
        control_action.joint_efforts = effort
        return control_action


@configclass
class SpotLegActuatorCfg(IdealPDActuatorCfg):
    """Configuration of :class:`SpotLegActuator`.

    ``effort_limit`` [N·m] clips the hips; give the knees ``inf``, their limit is the lookup.
    """

    class_type: type = SpotLegActuator

    knee_joint_expr: str = ".*_knee"
    """Joints that use the angle-dependent torque limit."""

    joint_parameter_lookup: list[list[float]] = MISSING
    """Knee lookup rows: joint angle [rad], transmission ratio, maximum torque [N·m]."""

    min_delay: int = 0
    """Minimum set-point delay [physics steps]."""

    max_delay: int = 0
    """Maximum set-point delay [physics steps]."""


def actuator_cfgs(physics_dt: float) -> dict:
    """The actuator groups the robot runs with (``robot.yaml`` ``actuators``): the fused layout of
    :func:`spot_actuator_cfgs` at the physics timestep ``physics_dt`` [s]."""
    return spot_actuator_cfgs("fused", physics_dt)


ACTUATOR_DELAY_S = (0.0, 0.008)
"""Leg set-point delay range [s], drawn per reset: 0–4 steps of the 2 ms training timestep."""


def actuator_delay_steps(physics_dt: float) -> tuple[int, int]:
    """Return :data:`ACTUATOR_DELAY_S` in physics steps at ``physics_dt`` [s], rounded."""
    return tuple(round(seconds / physics_dt) for seconds in ACTUATOR_DELAY_S)


def spot_actuator_cfgs(groups: str, physics_dt: float) -> dict:
    """Return Spot's actuator groups for the physics timestep ``physics_dt`` [s].

    The two layouts compute identical torques. Leg delays keep :data:`ACTUATOR_DELAY_S`.

    ``"fused"``: :class:`SpotLegActuator` for the twelve
    legs and one implicit group for the arm and gripper (``spot_legs``, ``arm_and_gripper``),
    half of IsaacLab's per-group cost per physics step. ``"isaaclab"``: IsaacLab's own classes
    (``spot_hip``, ``spot_knee``, ``arm_hold``, ``gripper_hold``), for example for
    Newton-native actuators, which translate only IsaacLab's classes.
    """
    min_delay, max_delay = actuator_delay_steps(physics_dt)
    hips = dict(effort_limit=45.0, stiffness=60.0, damping=1.5, min_delay=min_delay, max_delay=max_delay)
    arm = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]
    # URDF arm_f1x limit [N·m] bounds the squeeze on a grasped object. PhysX (TGS) leaks the squeeze
    # into the arm (the real squeeze is internal to the hand), holding arm_wr0 off target by about
    # squeeze torque x physics step, up to ~0.042 rad: at 250 Hz that cap is reached with this drive
    # and with half of it. With half (60, 10) the robot fell
    # stopping from 1 m/s and did not stop backwards (world lifecycle, footprint checks); the USD's
    # 120 / 20 passes both (2026-10-17, output_check/20261017-1443_stop_fall_crosscheck).
    hold = dict(
        arm=dict(effort_limit_sim=300.0, stiffness=400.0, damping=60.0),
        gripper=dict(effort_limit_sim=15.32, stiffness=120.0, damping=20.0),
    )
    if groups == "isaaclab":
        return {
            "spot_hip": DelayedPDActuatorCfg(joint_names_expr=[".*_hip_[xy]"], **hips),
            "spot_knee": RemotizedPDActuatorCfg(
                joint_names_expr=[".*_knee"],
                joint_parameter_lookup=joint_parameter_lookup,
                effort_limit=None,
                stiffness=hips["stiffness"],
                damping=hips["damping"],
                min_delay=hips["min_delay"],
                max_delay=hips["max_delay"],
            ),
            "arm_hold": ImplicitActuatorCfg(joint_names_expr=arm, **hold["arm"]),
            "gripper_hold": ImplicitActuatorCfg(joint_names_expr=["arm_f1x"], **hold["gripper"]),
        }
    if groups != "fused":
        raise ValueError(f"Unknown Spot actuator groups: {groups}")
    joints = {"arm": "arm_(sh|el|wr).*", "gripper": "arm_f1x"}
    return {
        "spot_legs": SpotLegActuatorCfg(
            joint_names_expr=[".*_hip_[xy]", ".*_knee"],
            # Knees: no box limit, as RemotizedPDActuator; the lookup limits them.
            effort_limit={".*_hip_[xy]": hips["effort_limit"], ".*_knee": float("inf")},
            stiffness=hips["stiffness"],
            damping=hips["damping"],
            joint_parameter_lookup=joint_parameter_lookup,
            min_delay=hips["min_delay"],
            max_delay=hips["max_delay"],
        ),
        "arm_and_gripper": ImplicitActuatorCfg(
            joint_names_expr=arm + ["arm_f1x"],
            **{key: {joints[part]: hold[part][key] for part in joints} for key in hold["arm"]},
        ),
    }
