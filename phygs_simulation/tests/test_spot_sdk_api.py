from __future__ import annotations

import time

import pytest

from skills.spot import (
    ImageClient,
    ImageResponse,
    ImageSource,
    LeaseClient,
    LeaseKeepAlive,
    RobotStateClient,
    create_standard_sdk,
    require_image_sources_ready,
    require_robot_state_ready,
)
from skills.spot.sdk import InMemoryLeaseBackend, InMemoryRobotControlBackend


_DEFAULT_STATE = object()


class _FakeRobotStateBackend:
    def __init__(self, state=_DEFAULT_STATE) -> None:
        self._state = {"powered": True} if state is _DEFAULT_STATE else state

    def get_robot_state(self, **kwargs):
        if self._state is None:
            return None
        if isinstance(self._state, dict):
            return {**self._state, "kwargs": kwargs}
        return self._state


class _FakeImageBackend:
    def __init__(self, images=None, sources=("frontleft", "hand")) -> None:
        self._images = images
        self._sources = tuple(sources)

    def list_image_sources(self):
        return tuple(ImageSource(name=name) for name in self._sources)

    def get_image_from_sources(self, image_sources):
        if self._images is not None:
            return tuple(ImageResponse(source=name, image=self._images[name]) for name in image_sources)
        return tuple(ImageResponse(source=name, image={"name": name}) for name in image_sources)


def test_sdk_create_robot_caches_by_address_and_preserves_name() -> None:
    sdk = create_standard_sdk("spot-sdk-test")

    robot = sdk.create_robot("sim://spot", name="spot-sim")

    assert sdk.create_robot("sim://spot") is robot
    with pytest.raises(RuntimeError):
        sdk.create_robot("sim://spot", name="other-name")


def test_robot_ensure_client_resolves_aliases_and_caches() -> None:
    sdk = create_standard_sdk("spot-sdk-test")
    robot = sdk.create_robot("sim://spot", name="spot-sim")

    state_backend = _FakeRobotStateBackend()
    image_backend = _FakeImageBackend()
    robot.install_service_factory("robot-state", lambda: RobotStateClient(state_backend))
    robot.install_service_factory("image", lambda: ImageClient(image_backend))
    robot.install_service_factory("manipulation", lambda: object())

    state_client = robot.ensure_client("robot_state")

    assert state_client is robot.ensure_client("robot-state")
    assert state_client.get_robot_state(requested_by="test")["kwargs"] == {"requested_by": "test"}
    assert robot.ensure_client("manipulation-api") is robot.ensure_client("manipulation")
    assert [source.name for source in robot.ensure_client("image").list_image_sources()] == ["frontleft", "hand"]


def test_require_robot_state_ready_returns_snapshot() -> None:
    state_client = RobotStateClient(_FakeRobotStateBackend({"powered": True}))

    snapshot = require_robot_state_ready(state_client)

    assert snapshot["powered"] is True


def test_require_robot_state_ready_fails_on_missing_snapshot() -> None:
    state_client = RobotStateClient(_FakeRobotStateBackend(None))

    with pytest.raises(RuntimeError, match="no state snapshot"):
        require_robot_state_ready(state_client)


def test_require_image_sources_ready_returns_depth_images() -> None:
    image_client = ImageClient(
        _FakeImageBackend(
            images={
                "frontleft": {"depth": [[1.0]]},
                "hand": {"depth": [[0.5]]},
            }
        )
    )

    responses = require_image_sources_ready(image_client, ("frontleft", "hand"))

    assert [response.source for response in responses] == ["frontleft", "hand"]


def test_require_image_sources_ready_fails_on_missing_source() -> None:
    image_client = ImageClient(_FakeImageBackend(images={"frontleft": {"depth": [[1.0]]}}, sources=("frontleft",)))

    with pytest.raises(RuntimeError, match="Missing required image sources"):
        require_image_sources_ready(image_client, ("frontleft", "hand"))


def test_require_image_sources_ready_fails_on_missing_depth() -> None:
    image_client = ImageClient(_FakeImageBackend(images={"frontleft": {"rgb": [[1]]}}, sources=("frontleft",)))

    with pytest.raises(RuntimeError, match="no depth frame"):
        require_image_sources_ready(image_client, ("frontleft",))


def test_power_on_requires_authentication_and_lease() -> None:
    sdk = create_standard_sdk("spot-sdk-test")
    robot = sdk.create_robot("sim://spot", name="spot-sim")
    lease_backend = InMemoryLeaseBackend()
    control_backend = InMemoryRobotControlBackend(client_name=robot.name, lease_backend=lease_backend)
    robot.install_control_backend(control_backend)
    robot.install_service_factory("lease", lambda: LeaseClient(lease_backend, client_name=robot.name))

    with pytest.raises(RuntimeError, match="authenticate"):
        robot.power_on()

    robot.authenticate("user", "password")
    with pytest.raises(RuntimeError, match="lease"):
        robot.power_on()

    lease_client = robot.ensure_client("lease")
    lease = lease_client.acquire()
    robot.power_on()

    assert robot.is_powered_on() is True

    robot.power_off()
    lease_client.return_lease(lease)

    assert robot.is_powered_on() is False


def test_lease_keep_alive_retains_and_returns_lease() -> None:
    backend = InMemoryLeaseBackend()
    lease_client = LeaseClient(backend, client_name="spot-sim")
    lease_client.acquire()

    with LeaseKeepAlive(
        lease_client,
        rpc_interval_seconds=0.01,
        return_at_exit=True,
    ):
        time.sleep(0.05)

    assert backend.retain_count >= 1
    assert backend.has_active_lease() is False
