from __future__ import annotations

import torch
import pytest

from skills.spot import (
    RobotCommandBuilder,
    RobotCommandClient,
    RobotCommandFeedback,
    RobotCommandFeedbackResponse,
    RobotCommandFeedbackStatus,
    StandCommandFeedback,
    StandCommandFeedbackStatus,
    SynchronizedCommandFeedback,
    MobilityCommandFeedback,
    blocking_command,
    blocking_stand,
)


class _FakeBackend:
    def __init__(self, feedback_sequence: list[RobotCommandFeedbackResponse]) -> None:
        self.feedback_sequence = feedback_sequence
        self.commands = []
        self.feedback_calls = 0
        self.command_id = 0

    def robot_command(self, command, end_time_secs=None) -> int:
        self.command_id += 1
        self.commands.append((command, end_time_secs))
        return self.command_id

    def robot_command_feedback(self, command_id: int) -> RobotCommandFeedbackResponse:
        index = min(self.feedback_calls, len(self.feedback_sequence) - 1)
        self.feedback_calls += 1
        response = self.feedback_sequence[index]
        return RobotCommandFeedbackResponse(command_id=command_id, feedback=response.feedback, message=response.message)


def _make_stand_feedback(status: StandCommandFeedbackStatus) -> RobotCommandFeedbackResponse:
    return RobotCommandFeedbackResponse(
        command_id=1,
        feedback=RobotCommandFeedback(
            status=RobotCommandFeedbackStatus.STATUS_PROCESSING,
            synchronized_feedback=SynchronizedCommandFeedback(
                mobility_command_feedback=MobilityCommandFeedback(
                    status=RobotCommandFeedbackStatus.STATUS_PROCESSING,
                    stand_feedback=StandCommandFeedback(status=status),
                )
            ),
        ),
    )


def test_build_synchro_command_combines_mobility_arm_and_gripper() -> None:
    mobility = RobotCommandBuilder.synchro_velocity_command(0.5, -0.1, 0.2)
    arm = RobotCommandBuilder.arm_joint_move_command(torch.tensor([1.0, 2.0]), ("j1", "j2"))
    gripper = RobotCommandBuilder.claw_gripper_open_angle_command(-0.3)

    combined = RobotCommandBuilder.build_synchro_command(mobility, arm, gripper)

    assert combined.mobility_command is not None
    assert combined.mobility_command.v_x == pytest.approx(0.5)
    assert combined.mobility_command.v_y == pytest.approx(-0.1)
    assert combined.mobility_command.v_rot == pytest.approx(0.2)
    assert combined.arm_command is not None
    assert tuple(combined.arm_command.joint_names) == ("j1", "j2")
    torch.testing.assert_close(combined.arm_command.positions, torch.tensor([1.0, 2.0]))
    assert combined.gripper_command is not None
    assert combined.gripper_command.position == pytest.approx(-0.3)


def test_blocking_stand_waits_for_standing_feedback() -> None:
    backend = _FakeBackend(
        [
            _make_stand_feedback(StandCommandFeedbackStatus.STATUS_IN_PROGRESS),
            _make_stand_feedback(StandCommandFeedbackStatus.STATUS_IS_STANDING),
        ]
    )
    client = RobotCommandClient(backend)

    blocking_stand(client, timeout_sec=0.1, update_frequency=100.0)

    assert len(backend.commands) == 1
    assert backend.commands[0][0].mobility_command is not None
    assert backend.feedback_calls >= 2


def test_blocking_command_times_out_when_status_never_matches() -> None:
    backend = _FakeBackend([_make_stand_feedback(StandCommandFeedbackStatus.STATUS_IN_PROGRESS)])
    client = RobotCommandClient(backend)
    command = RobotCommandBuilder.synchro_stand_command()

    with pytest.raises(TimeoutError):
        blocking_command(
            client,
            command,
            lambda response: response.feedback.synchronized_feedback.mobility_command_feedback.stand_feedback.status
            == StandCommandFeedbackStatus.STATUS_IS_STANDING,
            timeout_sec=0.02,
            update_frequency=100.0,
        )
