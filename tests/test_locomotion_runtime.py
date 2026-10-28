# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy write-scope regressions and runtime-backed command behavior."""

from types import SimpleNamespace

import pytest
import torch
import yaml
from fakes import SPOT, FakeRobot, FakeScene, FakeWorld, isaaclab_world

from phygs.execution.locomotion.base import DeadBand, Deceleration
from phygs.execution.locomotion.controller import LocomotionController
from phygs.robots.spot.policy import SpotFlatTerrainPolicy
from phygs.runtime.episode import EpisodeRuntime, JointTargets, RunMode
from phygs.skills.locomotion.api import LocomotionClient

NO_DEAD_BAND = DeadBand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
DECELERATION = Deceleration((2.0, 1.0, 2.0))
ARM = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")
TRAINING_ARM_DEFAULT = torch.tensor([0.0, -3.13, 3.13, 0.0, 0.0, 0.0])  # spot_env.yaml.
# The deployed locomotion configuration, so a recalibration does not break these tests.
LOCOMOTION = yaml.safe_load(SPOT.base.config.read_text())


def make_controller(policy, **kwargs):
    return LocomotionController(
        policy, dead_band=kwargs.pop("dead_band", NO_DEAD_BAND), deceleration=DECELERATION, stand_seconds=1.6, **kwargs
    )


class Robot(FakeRobot):
    num_joints = 19
    joint_names = tuple(f"joint_{index}" for index in range(19))  # Legs 0-11, arm 12-17 (ARM), gripper 18.
    body_names = ("body", "fl_foot", "fr_foot", "hl_foot", "hr_foot")

    def __init__(self):
        super().__init__(self.joint_names, position=0.2, velocity=0.3)

        def data(values):
            return SimpleNamespace(torch=torch.tensor([values], dtype=torch.float32))

        self.data.default_joint_pos = data([0.1] * 19)
        self.data.root_lin_vel_b = data([1, 2, 3])
        self.data.root_ang_vel_b = data([4, 5, 6])
        self.data.projected_gravity_b = data([0, 0, -1])
        self.data.body_pos_w = SimpleNamespace(torch=torch.zeros(1, 5, 3))  # All feet down.
        self.actuators = {
            "legs": SimpleNamespace(
                is_implicit_model=False,
                joint_indices=list(range(12)),
                stiffness=torch.full((1, 12), 60.0),
                damping=torch.full((1, 12), 1.5),
            )
        }

    def find_joints(self, patterns, preserve_order):
        if tuple(patterns) == ARM:
            return list(range(12, 18)), ARM
        return list(range(12)), self.joint_names[:12]


class Network:
    def __init__(self):
        self.observations = []

    def eval(self):
        return self

    def __call__(self, observation):
        self.observations.append(observation.clone())
        return torch.full((1, 12), 0.5)


@pytest.fixture
def policy(monkeypatch):
    network = Network()
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: network)
    policy = SpotFlatTerrainPolicy(SPOT.base.config, physics_dt=0.004)  # These tests step a 4 ms (250 Hz) fake world.
    policy.initialize(Robot())
    return policy


def test_compute_preserves_observation_and_decimation_without_writes(policy):
    for _ in range(11):
        policy.compute_targets(0.004, (0.4, 0.0, 0.0))
    assert policy.robot.writes == []
    assert len(policy.policy.observations) == 3
    observation = policy.policy.observations[0]
    assert observation.shape == (1, 60)
    torch.testing.assert_close(observation[0, :12], torch.tensor([1, 2, 3, 4, 5, 6, 0, 0, -1, 0.4, 0, 0]))
    torch.testing.assert_close(observation[0, 12:24], torch.full((12,), 0.1))
    torch.testing.assert_close(observation[0, 24:36], torch.full((12,), 0.3))
    torch.testing.assert_close(observation[0, 36:48], torch.zeros(12))
    # The arm, relative to its training default (not the robot's default), and its velocity.
    torch.testing.assert_close(observation[0, 48:54], 0.2 - TRAINING_ARM_DEFAULT)
    torch.testing.assert_close(observation[0, 54:], torch.full((6,), 0.3))
    torch.testing.assert_close(policy.policy.observations[1][0, 36:48], torch.full((12,), 0.5))
    assert policy.inference_dt == 0.02
    assert torch.all(policy.robot.actuators["legs"].stiffness == 60)


def test_reset_clears_policy_memory_and_restarts_inference(policy):
    policy.compute_targets(0.004, (0.4, 0, 0))
    policy.reset()
    policy.compute_targets(0.004, (0.4, 0, 0))
    assert len(policy.policy.observations) == 2
    assert torch.count_nonzero(policy.policy.observations[-1][0, 36:48]) == 0


def test_wrong_timing_and_nonfinite_commands_fail(policy):
    with pytest.raises(ValueError, match="physics_dt"):
        policy.compute_targets(1 / 240, (0.4, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        policy.compute_targets(0.004, (float("nan"), 0, 0))


def robot_world():
    """The robot's joints at 0.1 rad, following their targets."""
    return FakeWorld(Robot.joint_names, initial=0.1)


def bind(policy, source=RunMode.AUTONOMOUS, mode=RunMode.AUTONOMOUS):
    world = robot_world()
    controller = make_controller(policy, source=source)
    runtime = EpisodeRuntime(world, (controller,), mode=mode)
    client = LocomotionClient(runtime, controller)
    runtime.reset()
    return world, controller, runtime, client


def test_expiry_runs_stop_before_releasing_legs(policy):
    world, controller, runtime, client = bind(policy)
    command = client.command_velocity(0.4, 0, 0, duration=0.02)
    for _ in range(5):
        runtime.step()
    runtime.step()
    assert client.feedback(command).status == "stopping"
    assert controller.command == (0, 0, 0)
    with pytest.raises(RuntimeError, match="already owned"):
        runtime.acquire("locomotion", duration_ticks=1)
    # The stop slows 0.4 m/s at 2 m/s² (0.2 s, 50 ticks), then keeps the legs 1.6 s (400 ticks).
    for _ in range(449):
        runtime.step()
    assert client.feedback(command).status == "stopping"
    runtime.step()
    assert client.feedback(command).status == "done"
    assert runtime.controller_state("locomotion") == "idle"
    hip_stiffness = LOCOMOTION["stand_drive"][250]["hip"][0]
    assert set(world.gains.values()) == {hip_stiffness}  # The PhysX stand drive (these fake legs are all hips).
    assert all(value == 0.1 for joint, value in world.state.items() if joint not in controller.declaration.joints)


def test_release_keeps_the_standing_targets_not_the_sagged_measured_legs(policy):
    world, controller, runtime, client = bind(policy)
    legs = controller.declaration.joints
    measured = world.positions
    # Under load the legs sit below their targets; the release must not re-target them there.
    world.positions = lambda: {**measured(), **dict.fromkeys(legs, 0.3)}
    command = client.command_velocity(0.4, 0, 0, duration=0.02)
    for _ in range(1000):
        runtime.step()
    assert client.feedback(command).status == "done" and policy.standing
    assert all(world.state[joint] == pytest.approx(0.2) for joint in legs)  # The stand pose.
    assert controller.hold_targets() == pytest.approx(dict.fromkeys(legs, 0.2))


def test_hold_targets_are_empty_while_the_policy_walks(policy):
    controller = make_controller(policy)
    policy.reset()
    policy.compute_targets(policy._dt, (0.4, 0.0, 0.0))
    assert not policy.standing and controller.hold_targets() == {}


def test_updates_do_not_restart_policy_and_stop_has_feedback(policy, caplog):
    caplog.set_level("INFO", logger="phygs.skills.locomotion")
    _, _, runtime, client = bind(policy)
    first = client.command_velocity(0.4, 0, 0)
    for _ in range(2):
        runtime.step()
    second = client.command_velocity(0.3, 0.1, 0)
    client.command_velocity(0.3, 0.1, 0)  # An update with the same velocity is not narrated again.
    second = client.command_velocity(0.3, 0.1, 0)
    for _ in range(2):
        runtime.step()
    assert len(policy.policy.observations) == 1
    assert client.feedback(first).status == "overridden"
    assert client.feedback(second).status == "active"
    stop = client.stop()
    assert [r.getMessage() for r in caplog.records] == [
        "command_velocity: v_x 0.40 m/s, v_y 0.00 m/s, v_rot 0.00 rad/s for 0.50 s; then slowing down",
        "command_velocity: v_x 0.30 m/s, v_y 0.10 m/s, v_rot 0.00 rad/s for 0.50 s; then slowing down",
        "stop: controlled stop requested",
    ]
    assert client.feedback(second).status == "overridden"
    assert client.feedback(stop).status == "stopping"
    runtime.emergency_stop("test")
    assert client.feedback(stop).status == "interrupted"
    with pytest.raises(RuntimeError, match="interrupted"):
        client.command_velocity(0.1, 0, 0)


def test_reset_cancels_old_command_and_allows_new_command(policy):
    _, _, runtime, client = bind(policy)
    first = client.command_velocity(0.4, 0, 0)
    runtime.step()
    runtime.reset()
    assert client.feedback(first).status == "cancelled"
    second = client.command_velocity(0.2, 0, 0)
    assert second > first
    runtime.step()
    assert torch.count_nonzero(policy.policy.observations[-1][0, 36:48]) == 0


def test_human_client_cannot_mutate_autonomous_command(policy):
    _, controller, _, client = bind(policy, source=RunMode.HUMAN)
    with pytest.raises(PermissionError):
        client.command_velocity(0.4, 0, 0)
    assert controller.command == (0, 0, 0)


def test_direct_motion_requires_explicit_navigation_handoff(policy):
    _, _, runtime, client = bind(policy)
    navigation = SimpleNamespace(active=True)
    client.bind_navigation(navigation)
    with pytest.raises(RuntimeError, match="already bound"):
        client.bind_navigation(SimpleNamespace(active=False))
    with pytest.raises(RuntimeError, match="Cancel navigation"):
        client.command_velocity(1, 0, 0)
    with pytest.raises(RuntimeError, match="controlled stop"):
        client.stop()
    assert runtime.controller_state("locomotion") == "idle"
    navigation.active = False
    assert client.command_velocity(1, 0, 0) == 1
    assert client.stop() == 2


@pytest.mark.parametrize("duration", [0, -1, 0.003, float("nan")])
def test_invalid_duration_does_not_acquire_legs(policy, duration):
    _, _, runtime, client = bind(policy)
    with pytest.raises(ValueError):
        client.command_velocity(0.4, 0, 0, duration=duration)
    assert runtime.controller_state("locomotion") == "idle"


def test_gain_output_cannot_escape_joint_ownership():
    with pytest.raises(ValueError, match="within owned joints"):
        JointTargets({"leg": 1}, {"arm": 800}, {"arm": 25}).validate(("leg",))


def test_reset_does_not_turn_physical_velocity_into_a_command():
    robot = Robot()
    scene = FakeScene(robot)
    # Restoring the snapshot leaves its physical joint velocity [rad/s] as the velocity target.
    scene.reset_to = lambda state: robot.set_joint_velocity_target_index(target=robot.data.joint_vel.torch)
    world = isaaclab_world(scene)
    world.restore()
    torch.testing.assert_close(robot.data.joint_vel.torch, torch.full((1, 19), 0.3))
    torch.testing.assert_close(robot.velocity_target, torch.zeros(1, 19))
    torch.testing.assert_close(robot.effort_target, torch.zeros(1, 19))


def test_arm_and_locomotion_clients_share_runtime_without_conflicts(policy):
    from phygs.robots.control import ArmClient, RobotControl
    from phygs.robots.posture import PostureController

    world = robot_world()
    locomotion = make_controller(policy)
    posture = PostureController(world.joint_names[12:18])
    runtime = EpisodeRuntime(world, (locomotion, posture), mode=RunMode.AUTONOMOUS)
    client = LocomotionClient(runtime, locomotion)
    arm = ArmClient(runtime, posture, world, dict.fromkeys(posture.declaration.joints, (-1.0, 1.0)))
    control = RobotControl(runtime, client, arm, SPOT)
    control.reset(settling_seconds=0.004)
    walking = client.command_velocity(0.4, 0, 0)
    moving_arm = arm.command_joint_positions({"joint_12": 0.3}, duration=0.1)
    for _ in range(60):
        control.step()
    assert world.state["joint_12"] == pytest.approx(0.3)
    assert arm.feedback(moving_arm).status == "done"
    assert client.feedback(walking).status == "active"
    with pytest.raises(ValueError, match="outside"):
        arm.command_joint_positions({"joint_12": 2.0})
    runtime.emergency_stop("operator")
    control.reset(settling_seconds=0.004)
    assert client.feedback(walking).status == "interrupted"
    assert arm.feedback(moving_arm).status == "done"


def test_gain_buffers_restore_to_initial_values():
    robot = Robot()

    world = isaaclab_world(FakeScene(robot))
    world.apply(JointTargets({"joint_0": 0.2}, {"joint_0": 800.0}, {"joint_0": 25.0}))
    assert robot.actuators["legs"].stiffness[0, 0] == 800
    world.restore()
    assert torch.all(robot.actuators["legs"].stiffness == 60)
    assert torch.all(robot.actuators["legs"].damping == 1.5)


def test_stop_contract_is_checked_before_episode():
    from phygs.runtime.episode import ControllerDeclaration

    controller = SimpleNamespace(declaration=ControllerDeclaration("bad", ("joint_0",), stop_ticks=1))
    with pytest.raises(TypeError, match="no stop method"):
        EpisodeRuntime(robot_world(), (controller,), mode=RunMode.AUTONOMOUS)


def test_policy_targets_and_gains_cover_legs_only(policy):
    controller = make_controller(policy)
    targets = controller.standing_targets()
    expected = set(policy.robot.joint_names[:12])
    assert set(targets.positions) == expected
    assert set(targets.drive_stiffness) == expected  # Standing: PhysX joint drives.
    assert set(targets.drive_damping) == expected
    assert targets.stiffness == {} and targets.damping == {}
    assert policy.robot.writes == []


def test_policy_rejects_incomplete_leg_resolution(monkeypatch):
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: Network())
    robot = Robot()
    robot.find_joints = lambda *args, **kwargs: ([0], [robot.joint_names[0]])
    with pytest.raises(RuntimeError, match="12 leg joints"):
        SpotFlatTerrainPolicy(SPOT.base.config, physics_dt=0.004).initialize(robot)


def test_world_reuses_unchanged_targets_and_invalidates_on_restore():
    robot = Robot()

    world = isaaclab_world(FakeScene(robot))
    world.apply({"joint_2": 0.2, "joint_0": 0.1})
    world.apply({"joint_0": 0.1, "joint_2": 0.2})
    assert len(robot.writes) == 1
    world.apply(JointTargets({"joint_0": 0.1}, {"joint_0": 800}, {"joint_0": 25}))
    assert robot.actuators["legs"].stiffness[0, 0] == 800
    assert len(robot.writes) == 1
    world.apply({"joint_0": 0.3})
    assert len(robot.writes) == 2
    assert robot.writes[-1][0] == [0]
    world.restore()
    world.apply({"joint_0": 0.3})
    assert len(robot.writes) == 3


def test_python_commands_do_not_require_device_reductions(policy, monkeypatch):
    policy.compute_targets(0.004, (0.4, 0, 0))

    def unexpected(*args, **kwargs):
        raise AssertionError("Python command validation must not synchronize the device")

    monkeypatch.setattr(torch, "isfinite", unexpected)
    monkeypatch.setattr(torch, "count_nonzero", unexpected)
    policy.compute_targets(0.004, (0.4, 0, 0))
    policy.compute_targets(0.004, (0, 0, 0))


def test_dead_band_grounds_small_commands_and_raises_the_rest_to_the_lowest_walking_speed():
    band = DeadBand((0.1, 0.1, 0.05), (0.2, 0.25, 0.1), (0.26, 0.12, 0.1))
    assert band.apply((0.05, -0.08, 0.03)) == (0.0, 0.0, 0.0)  # Within zero: the robot stands.
    assert band.apply((0.15, -0.15, -0.07)) == (0.26, -0.25, -0.1)  # Raised, sign kept.
    assert band.apply((0.15, 0.11, 0.07)) == (0.26, 0.12, 0.1)
    assert band.apply((0.5, -0.4, 1.0)) == (0.5, -0.4, 1.0)  # Walking speeds pass unchanged.
    assert band.apply((0.1, -0.1, 0.05)) == (0.0, 0.0, 0.0)  # At zero: grounded.
    with pytest.raises(ValueError, match="zero band"):
        DeadBand((0.3, 0.1, 0.0), (0.2, 0.2, 0.0), (0.2, 0.2, 0.0))


def test_policy_receives_the_command_after_the_dead_band(policy):
    band = DeadBand.load(SPOT.base.config)  # The Spot configuration.
    controller = make_controller(policy, dead_band=band)
    received = []
    policy.compute_targets = lambda dt, command: received.append(command) or torch.zeros(1, 12)
    controller.command = (band.zero[0] / 2, 0.0, 0.0)
    controller.step(policy.control_dt)
    controller.command = ((band.zero[0] + band.lowest_positive[0]) / 2, 0.0, 0.0)
    controller.step(policy.control_dt)
    assert received == [(0.0, 0.0, 0.0), (band.lowest_positive[0], 0.0, 0.0)]


def test_deceleration_limits_slowing_but_not_speeding_up():
    limit = Deceleration((2.0, 1.0, 2.0))
    dt = 0.01
    assert limit.apply((0.0, 0.0, 0.0), (1.0, -0.8, 0.5), dt) == (1.0, -0.8, 0.5)  # Speeding up: at once.
    assert limit.apply((1.0, -0.8, 0.5), (0.0, 0.0, 0.0), dt) == pytest.approx((0.98, -0.79, 0.48))
    assert limit.apply((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), dt) == (0.0, 0.0, 0.0)  # Within one step: lands on it.
    assert limit.apply((0.5, 0.0, 0.0), (-0.5, 0.0, 0.0), dt) == pytest.approx((0.48, 0.0, 0.0))  # Reversing slows.
    assert limit.ramp_seconds((1.0, -0.8, 0.5)) == pytest.approx(0.8)  # Sideways: 0.8 m/s at 1 m/s².
    with pytest.raises(ValueError, match="positive"):
        Deceleration((2.0, 0.0, 2.0))


def test_policy_command_ramps_down_and_the_stop_interval_covers_the_ramp(policy):
    controller = make_controller(policy)
    received = []
    policy.compute_targets = lambda dt, command: received.append(command) or torch.zeros(1, 12)
    controller.command = (0.0, 0.6, 0.0)
    controller.step(0.004)
    controller.stop()
    assert controller.stop_duration_ticks() == 150 + 400  # 0.6 m/s at 1 m/s²: 0.6 s, then 1.6 s.
    controller.step(0.004)
    assert received == [(0.0, 0.6, 0.0), pytest.approx((0.0, 0.596, 0.0))]
    assert controller.sent_command == (0.0, 0.0, 0.0) and controller.applied_command == pytest.approx((0.0, 0.596, 0.0))


def test_stop_must_outlast_the_latest_stand_switch(policy):
    with pytest.raises(ValueError, match="outlast the latest stand switch"):
        LocomotionController(
            policy, dead_band=NO_DEAD_BAND, deceleration=DECELERATION, stand_seconds=policy.stand_switch_seconds
        )


def test_stand_waits_for_the_feet_to_be_down(policy):
    feet = policy.robot.data.body_pos_w.torch
    feet[0, 1, 2] = 0.05  # A foot in swing.
    policy.compute_targets(0.004, (0.4, 0, 0))
    ticks = 0
    while not policy.standing:
        policy.compute_targets(0.004, (0, 0, 0))
        ticks += 1
    assert ticks * 0.004 == pytest.approx(policy.stand_switch_seconds)  # Forced at the latest.
    policy.compute_targets(0.004, (0.4, 0, 0))
    feet[0, 1, 2] = 0.01  # Down within feet_down_spread, still moving fast: from feet_down_after.
    ticks = 0
    while not policy.standing:
        policy.compute_targets(0.004, (0, 0, 0))
        ticks += 1
    assert ticks * 0.004 == pytest.approx(LOCOMOTION["stand_switch"]["feet_down_after"])


def test_world_hands_joints_to_physx_drives_and_back():
    robot = Robot()
    legs = robot.actuators["legs"]
    legs.driven = set()
    legs.set_physx_drive = lambda joints, on: (legs.driven.update if on else legs.driven.difference_update)(joints)
    written = {}
    robot.write_joint_stiffness_to_sim_index = lambda *, stiffness, joint_ids: written.update(
        {("kp", j): float(v) for j, v in zip(joint_ids, stiffness[0])}
    )
    robot.write_joint_damping_to_sim_index = lambda *, damping, joint_ids: written.update(
        {("kd", j): float(v) for j, v in zip(joint_ids, damping[0])}
    )

    world = isaaclab_world(FakeScene(robot))
    world.apply(JointTargets({"joint_0": 0.2, "joint_1": 0.2}, drive_stiffness={"joint_0": 2000.0, "joint_1": 2500.0},
                             drive_damping={"joint_0": 80.0, "joint_1": 80.0}))  # fmt: skip
    assert legs.driven == {0, 1} and written == {("kp", 0): 2000, ("kp", 1): 2500, ("kd", 0): 80, ("kd", 1): 80}
    assert torch.all(legs.stiffness == 60)  # The explicit gains stay for walking.
    world.apply(JointTargets({"joint_0": 0.2}, {"joint_0": 60.0}, {"joint_0": 1.5}))  # Explicit gains: back.
    assert legs.driven == {1} and written[("kp", 0)] == 0 and written[("kd", 0)] == 0
    world.restore()
    assert legs.driven == set() and written[("kp", 1)] == 0
    with pytest.raises(ValueError, match="either explicit or PhysX"):
        JointTargets({"joint_0": 0.2}, {"joint_0": 1.0}, {"joint_0": 1.0}, drive_stiffness={"joint_0": 1.0},
                     drive_damping={"joint_0": 1.0}).validate(("joint_0",))  # fmt: skip
