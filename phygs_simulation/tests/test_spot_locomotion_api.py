from __future__ import annotations

from skills.locomotion import SpotLocomotionClient


class _FakePolicy:
    def __init__(self) -> None:
        self._dt = 0.01
        self.forward_calls: list[tuple[float, tuple[float, float, float]]] = []

    def forward(self, dt: float, command) -> None:
        self.forward_calls.append((dt, tuple(float(value) for value in command.tolist())))


def test_spot_locomotion_client_tracks_current_spot_command() -> None:
    policy = _FakePolicy()
    locomotion = SpotLocomotionClient(policy=policy)

    locomotion.command_velocity(0.5, -0.2, 0.3)
    locomotion.step()
    locomotion.stop()
    locomotion.step()

    assert policy.forward_calls[0][0] == policy._dt
    assert policy.forward_calls[0][1] == (0.5, -0.20000000298023224, 0.30000001192092896)
    assert policy.forward_calls[1] == (policy._dt, (0.0, 0.0, 0.0))
    assert locomotion.mode == "stand"
    assert locomotion.current_command == (0.0, 0.0, 0.0)
