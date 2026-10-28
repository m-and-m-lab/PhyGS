# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot's fused leg actuator against IsaacLab's hip and knee actuators, bit for bit."""

import itertools

import pytest
import torch
from isaaclab.utils.types import ArticulationActions

from phygs.robots.description import load_robot
from phygs.robots.spot.actuators import actuator_delay_steps, spot_actuator_cfgs
from phygs.robots.spot.policy import SpotFlatTerrainPolicy

# Spot-like joint order: legs interleaved with the arm, hips and knees mixed.
JOINTS = (
    "arm_sh0", "fl_hip_x", "fr_hip_x", "hl_hip_x", "hr_hip_x", "arm_sh1", "fl_hip_y", "fr_hip_y", "hl_hip_y",
    "hr_hip_y", "arm_el0", "fl_knee", "fr_knee", "hl_knee", "hr_knee", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x",
)  # fmt: skip
ENVS = 3


def build(cfgs: dict) -> dict:
    """Instantiate actuator groups as IsaacLab's articulation does, from the same joint defaults."""
    import re

    out = {}
    for name, cfg in cfgs.items():
        ids = [i for i, joint in enumerate(JOINTS) if any(re.fullmatch(e, joint) for e in cfg.joint_names_expr)]
        n = len(ids)
        out[name] = cfg.class_type(
            cfg, joint_names=[JOINTS[i] for i in ids], joint_ids=torch.tensor(ids, dtype=torch.int32),
            num_envs=ENVS, device="cpu", stiffness=torch.zeros(ENVS, n), damping=torch.zeros(ENVS, n),
            effort_limit=torch.full((ENVS, n), 1.0e9), velocity_limit=torch.full((ENVS, n), 12.0),
        )  # fmt: skip
    return out


def per_joint(groups: dict, attribute: str) -> dict:
    """Map each joint to its value of ``attribute`` in the environment 0."""
    values = {}
    for actuator in groups.values():
        for column, joint in enumerate(actuator.joint_names):
            values[joint] = float(getattr(actuator, attribute)[0, column])
    return values


@pytest.fixture
def layouts():
    # At 4 ms the 0-8 ms delays are 0-2 steps: every lag pair the layouts must agree on.
    return build(spot_actuator_cfgs("isaaclab", 0.004)), build(spot_actuator_cfgs("fused", 0.004))


def test_both_layouts_configure_every_joint_alike(layouts):
    lab, fused = layouts
    assert set(fused) == {"spot_legs", "arm_and_gripper"}
    for attribute in ("stiffness", "damping", "effort_limit", "effort_limit_sim", "velocity_limit",
                      "velocity_limit_sim", "armature", "friction"):  # fmt: skip
        assert per_joint(lab, attribute) == per_joint(fused, attribute), attribute
    assert per_joint(fused, "effort_limit")["fl_knee"] == float("inf")  # Knees: the lookup limits them.


def step_inputs(generator: torch.Generator, legs: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    """Random set-points and states; knee angles also beyond the lookup table [rad]."""
    n = len(legs)
    knee = torch.tensor(["knee" in joint for joint in legs])
    pos = torch.rand(ENVS, n, generator=generator) * 2.0 - 1.0
    pos[:, knee] = torch.rand(ENVS, int(knee.sum()), generator=generator) * -3.4 + 0.3
    targets = pos + torch.randn(ENVS, n, generator=generator) * 0.3
    return (targets, torch.randn(ENVS, n, generator=generator), torch.randn(ENVS, n, generator=generator) * 5.0,
            pos, torch.randn(ENVS, n, generator=generator) * 4.0)  # fmt: skip


def run_both(lab: dict, fused: dict, lags: tuple[int, int] | None, seed: int = 0) -> None:
    """Step both layouts on identical inputs and require identical torques every step."""
    hip, knee, legs = lab["spot_hip"], lab["spot_knee"], fused["spot_legs"]
    columns = {joint: c for c, joint in enumerate(legs.joint_names)}
    hip_cols = torch.tensor([columns[j] for j in hip.joint_names])
    knee_cols = torch.tensor([columns[j] for j in knee.joint_names])

    def reset(env_ids=None):
        torch.manual_seed(seed)
        hip.reset(env_ids)
        knee.reset(env_ids)
        torch.manual_seed(seed)
        legs.reset(env_ids)
        if lags is not None:
            for actuator, lag in ((hip, lags[0]), (knee, lags[1])):
                for buffer in (actuator.positions_delay_buffer, actuator.velocities_delay_buffer,
                               actuator.efforts_delay_buffer):  # fmt: skip
                    buffer.set_time_lag(lag)
            legs.set_time_lags(*lags)
        # The same random draw order: hips, then knees.
        for actuator, cols in ((hip, hip_cols), (knee, knee_cols)):
            drawn = actuator.positions_delay_buffer.time_lags.long().unsqueeze(1).expand(-1, len(cols))
            assert torch.equal(legs.time_lags[:, cols], drawn)

    generator = torch.Generator().manual_seed(seed)
    reset()
    for step in range(40):
        if step == 15:  # A partial reset of one environment.
            reset([1])
        if step == 25:  # Standing gains, as the locomotion policy switches them.
            for actuator in (hip, knee, legs):
                actuator.stiffness[:] = 800.0
                actuator.damping[:] = 25.0
        targets, velocities, efforts, pos, vel = step_inputs(generator, legs.joint_names)
        expected = torch.empty(ENVS, len(legs.joint_names))
        computed = torch.empty_like(expected)
        for actuator, cols in ((hip, hip_cols), (knee, knee_cols)):
            out = actuator.compute(
                ArticulationActions(targets[:, cols].clone(), velocities[:, cols].clone(), efforts[:, cols].clone()),
                pos[:, cols],
                vel[:, cols],
            )
            expected[:, cols] = out.joint_efforts
            computed[:, cols] = actuator.computed_effort
        out = legs.compute(ArticulationActions(targets.clone(), velocities.clone(), efforts.clone()), pos, vel)
        assert torch.equal(out.joint_efforts, expected), (step, (out.joint_efforts - expected).abs().max())
        assert torch.equal(legs.computed_effort, computed), step
        assert torch.equal(legs.applied_effort, expected), step


@pytest.mark.parametrize("lags", list(itertools.product(range(3), range(3))))
def test_fused_legs_equal_isaaclab_for_every_delay(layouts, lags):
    run_both(*layouts, lags)


@pytest.mark.parametrize("seed", range(4))
def test_fused_legs_draw_the_same_random_delays(layouts, seed):
    run_both(*layouts, None, seed)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="Unknown Spot actuator groups"):
        spot_actuator_cfgs("custom", 0.004)


@pytest.mark.parametrize(
    ("physics_dt", "delay_steps"), [(0.002, (0, 4)), (0.004, (0, 2)), (0.005, (0, 2)), (0.01, (0, 1))]
)
def test_leg_delays_keep_their_time_at_other_physics_rates(physics_dt, delay_steps):
    assert actuator_delay_steps(physics_dt) == delay_steps
    for groups in ("fused", "isaaclab"):
        legs = [cfg for cfg in spot_actuator_cfgs(groups, physics_dt).values() if hasattr(cfg, "max_delay")]
        assert len(legs) == (1 if groups == "fused" else 2)
        assert {(cfg.min_delay, cfg.max_delay) for cfg in legs} == {delay_steps}


def test_policy_keeps_its_50_hz_period_at_the_250_hz_physics_rate():
    policy = SpotFlatTerrainPolicy(load_robot("spot").base.config, physics_dt=0.004)
    assert policy.control_dt == 0.004
    assert policy.training_dt == 0.002  # Trained at 500 Hz; deployed at 250 Hz.
    assert policy.inference_dt == pytest.approx(0.02)  # The trained 50 Hz policy period.


def test_policy_rejects_a_timestep_that_does_not_divide_its_period():
    with pytest.raises(ValueError, match="does not divide the policy period"):
        SpotFlatTerrainPolicy(load_robot("spot").base.config, physics_dt=0.003)


def test_policy_rejects_a_physics_rate_without_stand_drive_gains():
    with pytest.raises(ValueError, match="No stand drive gains for 100 Hz"):
        SpotFlatTerrainPolicy(load_robot("spot").base.config, physics_dt=0.01)
    with pytest.raises(ValueError, match="No stand drive gains for 500 Hz"):
        SpotFlatTerrainPolicy(load_robot("spot").base.config, physics_dt=0.002)


def test_physx_drive_hands_joints_their_undelayed_set_point_and_no_effort(layouts):
    _, fused = layouts
    legs = fused["spot_legs"]
    legs.reset(None)
    legs.set_time_lags(4, 4)
    generator = torch.Generator().manual_seed(0)
    knees = [c for c, joint in enumerate(legs.joint_names) if joint.endswith("_knee")]
    legs.set_physx_drive(knees, True)
    for _ in range(3):
        targets, velocities, efforts, pos, vel = step_inputs(generator, legs.joint_names)
        out = legs.compute(ArticulationActions(targets.clone(), velocities.clone(), efforts.clone()), pos, vel)
    assert torch.equal(legs.physx_drive, torch.tensor(["_knee" in j for j in legs.joint_names]))
    assert torch.all(out.joint_efforts[:, knees] == 0) and torch.all(legs.computed_effort[:, knees] == 0)
    assert torch.equal(out.joint_positions, targets) and torch.all(out.joint_velocities == 0)  # Undelayed.
    assert torch.all(out.joint_efforts[:, [c for c in range(12) if c not in knees]] != 0)  # Hips: delayed PD.
    legs.set_physx_drive(knees, False)
    out = legs.compute(ArticulationActions(targets.clone(), velocities.clone(), efforts.clone()), pos, vel)
    assert out.joint_positions is None and torch.all(out.joint_efforts[:, knees] != 0)
