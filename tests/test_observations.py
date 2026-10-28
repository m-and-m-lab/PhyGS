# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation permissions, scheduling, capture-time transforms, and reset isolation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from fakes import FakeRobot, FakeScene, FakeWorld, ObservationScene, array, isaaclab_world, observations

from phygs.observations.types import CameraPermission, ObservationConfig, Stamp
from phygs.runtime.episode import EpisodeRuntime, EpisodeStatus, RunMode
from phygs.simulation.observations import IsaacLabObservations, pose_matrix


def source(scene=None, **kwargs):
    scene = ObservationScene() if scene is None else scene
    kwargs.setdefault("camera_period", 0.1)
    return scene, observations(scene, ObservationConfig(cameras=(CameraPermission("front"),), **kwargs))


def test_restricted_products_do_not_expose_world_or_unpermitted_images():
    scene = ObservationScene()
    del scene["front"].data.output["rgb"]
    adapter = observations(
        scene, ObservationConfig(joint_state=False, imu=False, cameras=(CameraPermission("front", rgb=False),))
    )
    adapter.reset(1)
    packet = adapter.observe()
    assert packet.localization is packet.joints is packet.imu is None
    assert packet.cameras["front"].rgb is None
    assert {(edge.target, edge.source) for edge in packet.transforms} == {("body", "front_optical")}
    with pytest.raises(KeyError):
        packet.transform("world", "front_optical", packet.stamp)
    torch.testing.assert_close(packet.cameras["front"].extrinsics.matrix[:3, 3], torch.tensor([0.5, 0, 0]))


def test_capture_rates_keep_old_images_and_exact_time_transforms():
    scene, adapter = source(localization="ground_truth")
    adapter.reset(1)
    initial = adapter.observe()
    scene["robot"].data.root_pose_w.torch[0, 0] += 1
    scene["front"].data.pos_w.torch[0, 0] += 1
    adapter.update(1, 10)  # 0.04 s: a state sample, no camera capture (every 0.1 s = 25 ticks).
    packet = adapter.observe()
    assert packet.joints.stamp.tick == 10
    assert packet.cameras["front"].stamp.tick == 0
    assert packet.stamp.time_seconds == pytest.approx(0.04)
    assert packet.transform("world", "front_optical", initial.stamp)[0, 3] == 4.5
    assert packet.localization.pose.matrix[0, 3] == 5
    with pytest.raises(KeyError):
        packet.transform("world", "front_optical", packet.stamp)
    adapter.update(1, 25)
    latest = adapter.observe()
    assert latest.cameras["front"].stamp.time_seconds == pytest.approx(0.1)
    assert latest.transform("world", "front_optical", latest.stamp)[0, 3] == 5.5
    torch.testing.assert_close(
        latest.transform("front_optical", "world", latest.stamp)
        @ latest.transform("world", "front_optical", latest.stamp),
        torch.eye(4),
    )


def test_snapshots_and_reads_cannot_modify_cached_or_sensor_data():
    scene, adapter = source()
    adapter.reset(1)
    packet = adapter.observe()
    packet.joints.positions.fill_(99)
    packet.cameras["front"].depth.fill_(99)
    packet.cameras.clear()
    again = adapter.observe()
    assert again.joints.positions[0] == pytest.approx(0.1)
    assert again.cameras["front"].depth[0, 0] == 2
    assert scene["front"].data.output["distance_to_image_plane"].torch[0, 0, 0, 0] == 2


def test_depth_unprojection_uses_optical_depth_and_masks_invalid_pixels():
    _, adapter = source()
    adapter.reset(1)
    camera = adapter.observe().cameras["front"]
    assert camera.depth_valid.tolist() == [[True, False]]
    torch.testing.assert_close(camera.points(), torch.tensor([[0, 0, 2.0]]))
    scene = ObservationScene()
    adapter = observations(scene, ObservationConfig(cameras=(CameraPermission("front", depth=False),)))
    adapter.reset(1)
    with pytest.raises(PermissionError):
        adapter.observe().cameras["front"].points()


def test_strided_unprojection_keeps_every_stride_th_row_and_column_at_its_own_pixel():
    _, adapter = source()
    adapter.reset(1)
    camera = adapter.observe().cameras["front"]
    depth = torch.full((4, 6), 2.0)
    valid = torch.ones((4, 6), dtype=torch.bool)
    valid[2, 4] = False
    camera = replace(camera, depth=depth, depth_valid=valid, intrinsics=torch.eye(3))
    points = camera.points(stride=2)
    # Rows 0 and 2, columns 0, 2 and 4; (row 2, column 4) is invalid. Identity intrinsics: x = column * z.
    assert points.shape == (5, 3)
    assert sorted((float(x) / 2, float(y) / 2) for x, y, _ in points) == [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0)]
    torch.testing.assert_close(camera.points(stride=1)[:, 2], torch.full((23,), 2.0))
    assert sorted({float(y) / 2 for _, y, _ in camera.points(rows=2)}) == [0, 1]  # The top two rows.
    with pytest.raises(ValueError, match="stride"):
        camera.points(stride=0)


def test_depth_nearer_than_the_minimum_is_no_depth_for_every_consumer():
    scene = ObservationScene()
    scene["front"].data.output["distance_to_image_plane"] = array([[[[0.15], [0.25]]]])  # [m]
    _, adapter = source(scene)
    adapter.reset(1)
    camera = adapter.observe().cameras["front"]
    assert camera.depth_valid.tolist() == [[False, True]]
    assert camera.depth.tolist() == [[0.0, pytest.approx(0.25)]]  # The rendered 0.15 m is not exposed.
    _, adapter = source(scene, depth_min=0.1)
    adapter.reset(1)
    assert adapter.observe().cameras["front"].depth_valid.tolist() == [[True, True]]
    with pytest.raises(ValueError, match="minimum depth"):
        ObservationConfig(depth_min=-0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_depth_from_the_gpu_is_masked_there_and_reaches_the_scene_device_unchanged():
    """The renderer delivers depth on the GPU; the masked depth and mask must equal the CPU path's."""
    scene = ObservationScene()
    scene["front"].data.output["distance_to_image_plane"] = array(
        [[[[0.15], [0.25]], [[float("inf")], [float("nan")]]]]
    )
    scene["front"].data.output["rgb"] = array([[[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [9, 9, 9]]]], torch.uint8)
    packets = []
    for device in ("cpu", "cuda"):
        adapter = observations(scene, ObservationConfig(cameras=(CameraPermission("front"),)), depth_device=device)
        adapter.reset(1)
        packets.append(adapter.observe().cameras["front"])
    cpu, gpu = packets
    assert gpu.depth.device.type == "cpu" and gpu.depth_valid.device.type == "cpu"
    assert torch.equal(gpu.depth, cpu.depth) and torch.equal(gpu.depth_valid, cpu.depth_valid)
    assert gpu.depth_valid.tolist() == [[False, True], [False, False]]


def test_reset_discards_old_episode_and_requires_initialization():
    _, adapter = source(localization="ground_truth")
    with pytest.raises(RuntimeError, match="Reset"):
        adapter.observe()
    adapter.reset(7)
    adapter.update(7, 50)
    old = adapter.observe()
    with pytest.raises(RuntimeError, match="Reset"):
        adapter.update(8, 0)
    adapter.reset(8)
    new = adapter.observe()
    assert new.stamp == Stamp(8, 0, 0.0)
    assert all(edge.stamp.episode_id == 8 for edge in new.transforms)
    with pytest.raises(KeyError, match="another episode"):
        new.transform("world", "body", old.stamp)


@pytest.mark.parametrize("kwargs", [{"state_period": 0.003}, {"camera_period": 0}, {"camera_period": float("nan")}])
def test_invalid_sample_periods_fail_before_episode(kwargs):
    with pytest.raises(ValueError):
        source(**kwargs)


def test_configuration_rejects_unknown_localization_and_missing_sensors():
    with pytest.raises(ValueError, match="Unsupported localization"):
        ObservationConfig(localization="estimated")
    with pytest.raises(ValueError, match="unique"):
        ObservationConfig(cameras=(CameraPermission("front"), CameraPermission("front")))
    with pytest.raises(ValueError, match="Missing configured camera"):
        observations(ObservationScene(), ObservationConfig(cameras=(CameraPermission("absent"),)))
    scene = ObservationScene()
    del scene.sensors["observation_imu"]
    with pytest.raises(ValueError, match="IMU permission"):
        IsaacLabObservations(scene, ObservationConfig(), 0.004, base=None)


def test_quaternion_order_and_transform_direction():
    matrix = pose_matrix(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]))
    torch.testing.assert_close(matrix @ torch.tensor([1.0, 0.0, 0.0, 1.0]), torch.tensor([1.0, 3.0, 3.0, 1.0]))


def test_runtime_owns_sampling_and_records_configuration():
    world = FakeWorld(("leg",))
    _, adapter = source(localization="ground_truth")
    runtime = EpisodeRuntime(world, (), mode=RunMode.AUTONOMOUS, observations=adapter)
    runtime.reset(settling_ticks=5)
    assert world.steps == 5 and runtime.observe().stamp.tick == 0
    runtime.step()
    assert world.steps == 6 and runtime.observe().stamp.tick == 1
    assert runtime.result()["observations"]["localization"] == "ground_truth"
    runtime.reset()
    assert runtime.observe().stamp.episode_id == 2
    assert runtime.observe().stamp.tick == 0


def test_capture_failure_terminates_runtime():
    scene, adapter = source()
    runtime = EpisodeRuntime(FakeWorld(("leg",)), (), mode=RunMode.AUTONOMOUS, observations=adapter)
    runtime.reset()
    scene["front"].data.intrinsic_matrices.torch.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="Invalid calibration"):
        for _ in range(50):
            runtime.step()
    assert runtime.status == EpisodeStatus.FAILED


def test_failed_reset_cannot_return_previous_episode(monkeypatch):
    world = FakeWorld(("leg",))
    _, adapter = source()
    runtime = EpisodeRuntime(world, (), mode=RunMode.AUTONOMOUS, observations=adapter)
    runtime.reset()

    def fail_restore():
        raise RuntimeError("bad scene")

    monkeypatch.setattr(world, "restore", fail_restore)
    with pytest.raises(RuntimeError, match="bad scene"):
        runtime.reset()
    with pytest.raises(RuntimeError, match="current episode"):
        runtime.observe()


def test_physics_sensors_update_every_step_including_settling():
    reads = []

    class Imu:
        @property
        def data(self):
            reads.append(scene.ticks)
            return None

    scene = FakeScene(FakeRobot(("leg",)), sensors={"imu": Imu()})
    world = isaaclab_world(scene, physics_sensors=("imu",))
    runtime = EpisodeRuntime(world, (), mode=RunMode.AUTONOMOUS)
    runtime.reset(settling_ticks=3)
    runtime.step()
    assert reads == [1, 2, 3, 4]


def test_default_camera_rate_and_capture_time_rendering():
    config = ObservationConfig(cameras=(CameraPermission("front"),))
    assert config.camera_period <= 1 / 30
    scene = ObservationScene()
    renders = []

    def render():
        renders.append(len(renders))
        scene["front"].data.output["rgb"].torch.fill_(len(renders))

    adapter = observations(scene, config, render=render)
    adapter.reset(1)
    adapter.update(1, 7)  # The default 0.032 s period is 8 ticks at 250 Hz.
    assert renders == [0]
    assert adapter.observe().cameras["front"].rgb[0, 0, 0] == 1
    adapter.update(1, 8)
    assert renders == [0, 1]
    assert adapter.observe().cameras["front"].rgb[0, 0, 0] == 2
    assert adapter.observe().cameras["front"].stamp.tick == 8


def test_fabric_pose_writes_only_in_the_step_before_each_capture():
    from phygs.simulation.fabric import FabricTransformWrites

    class Writes:
        enabled = True
        before_capture = FabricTransformWrites.before_capture

        def set(self, enabled):
            self.enabled = enabled

    writes = Writes()
    adapter = observations(
        ObservationScene(), ObservationConfig(cameras=(CameraPermission("front"),)), fabric_writes=writes
    )
    adapter.reset(1)  # Captures at tick 0, then every 8 ticks (0.032 s at 250 Hz).
    written = []
    for tick in range(1, 33):
        written.append(writes.enabled)  # Setting in effect for the step to this tick.
        adapter.update(1, tick)
    # The camera, a child of the robot body, follows the body's write one step late.
    assert [tick for tick, on in enumerate(written, 1) if on] == [7, 8, 15, 16, 23, 24, 31, 32]
    assert not FabricTransformWrites.applies(SimpleNamespace(has_gui=False), cameras=False)
    viewer = SimpleNamespace(has_gui=True, has_active_visualizers=lambda: False)
    assert not FabricTransformWrites.applies(viewer, cameras=True)
    visualizer = SimpleNamespace(has_gui=False, has_active_visualizers=lambda: True)
    assert not FabricTransformWrites.applies(visualizer, cameras=True)
    headless = SimpleNamespace(has_gui=False, has_active_visualizers=lambda: False)
    assert FabricTransformWrites.applies(headless, cameras=True)
