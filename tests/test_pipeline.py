# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Information boundaries and optional processing across agent arrangements."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from fakes import SPOT_COVERAGE, SPOT_DEAD_BAND, FakeWorld, Motion, ObservationScene, Runtime, grid, observations

from phygs.execution.navigation.backends.nav2 import (
    Nav2BackendFactory,
    Nav2Config,
    Nav2NavigationBackend,
    Nav2Update,
)
from phygs.observations.types import (
    CameraPermission,
    LocalizationObservation,
    ObservationAccess,
    ObservationConfig,
    Stamp,
    Transform,
)
from phygs.runtime.pipeline import (
    ObservationPipeline,
    PipelineConfig,
    ProcessingService,
    ServiceDeclaration,
)
from phygs.skills.navigation.api import NavigationClient, NavigationConfig


def capture(localization="ground_truth"):
    return ObservationConfig(localization=localization, cameras=(CameraPermission("front"),), camera_period=0.1)


def make_pipeline(config, **kwargs):
    scene = ObservationScene()
    scene["robot"].data.root_pose_w.torch[0, :3] = torch.tensor([0.0, 0.0, 0.5])
    scene["front"].data.pos_w.torch[0] = torch.tensor([0.5, 0.0, 0.5])
    pipeline = ObservationPipeline(observations(scene, config.capture), config, **kwargs)
    pipeline.reset(1)
    return pipeline


def test_agent_images_do_not_leak_private_localization_through_transform_graph():
    config = capture()
    agent = ObservationAccess(cameras=(CameraPermission("front", depth=False),))
    nav = ServiceDeclaration("navigation", ObservationAccess(joint_state=True, localization="ground_truth"))
    pipeline = make_pipeline(PipelineConfig(config, agent, ObservationAccess.from_config(config)), consumers=(nav,))
    packet = pipeline.observe()
    assert packet.localization is packet.joints is packet.imu is None
    assert packet.cameras["front"].depth is None
    assert len(packet.transforms) == 1
    with pytest.raises(KeyError):
        packet.transform("world", "front_optical", packet.cameras["front"].stamp)
    private = pipeline.view("navigation").observe()
    assert private.localization.mode == "ground_truth"
    assert not private.cameras
    packet.cameras["front"].extrinsics.matrix.fill_(99)
    assert pipeline.observe().cameras["front"].extrinsics.matrix[3, 3] == 1
    sources = pipeline.configuration["services"][0]["effective_information_sources"]
    assert "ground_truth_localization" in sources


def test_pipeline_shares_trusted_capture_but_detaches_agent_images():
    pipeline = make_pipeline(PipelineConfig.localized(capture()))
    internal = pipeline._source.observe_internal()
    assert pipeline._packet is internal
    public = pipeline.observe()
    public.cameras["front"].rgb.fill_(0)
    assert torch.any(internal.cameras["front"].rgb != 0)
    assert torch.any(pipeline.observe().cameras["front"].rgb != 0)
    pipeline.update(1, 1)
    assert pipeline._packet is pipeline._source.observe_internal()


def test_task_permissions_apply_to_indirect_assistance_before_capture():
    config = capture()
    raw = replace(ObservationAccess.from_config(config), localization="none")
    with pytest.raises(PermissionError, match="capture"):
        ObservationPipeline.validate(PipelineConfig(config, raw, raw))
    known = ServiceDeclaration("navigation", raw, privileged_sources=("privileged_known_box_geometry",))
    with pytest.raises(PermissionError, match="indirect"):
        ObservationPipeline.validate(PipelineConfig.observations_only(config), consumers=(known,))
    depth = ServiceDeclaration("mapper", ObservationAccess(cameras=(CameraPermission("front", rgb=False),)))
    rgb = replace(config, cameras=(CameraPermission("front", depth=False),))
    with pytest.raises(PermissionError, match="camera"):
        ObservationPipeline.validate(PipelineConfig.localized(rgb), consumers=(depth,))


def test_raw_and_localized_presets_do_not_create_mapping_or_navigation():
    for config in (PipelineConfig.observations_only(capture()), PipelineConfig.localized(capture())):
        pipeline = make_pipeline(config)
        assert pipeline.configuration["services"] == []
        assert (pipeline.observe().localization is None) == (config.agent.localization == "none")
        with pytest.raises(KeyError):
            pipeline.product("mapping")
        with pytest.raises(KeyError):
            pipeline.view("navigation")


def test_processor_is_optional_reset_isolated_and_not_updated_by_reads():
    config = capture()
    mapper = Integrator()
    declaration = ServiceDeclaration(
        "mapping",
        ObservationAccess(
            localization="ground_truth",
            cameras=(CameraPermission("front", rgb=False),),
        ),
    )
    service = ProcessingService(declaration, mapper, expose_to_agent=True)
    pipeline = make_pipeline(PipelineConfig.localized(config), processors=(service,))
    assert pipeline.product("mapping").frames_integrated == 1
    pipeline.observe()
    pipeline.product("mapping")
    assert mapper.snapshot().frames_integrated == 1
    pipeline.update(1, 10)
    assert mapper.snapshot().frames_integrated == 1
    pipeline.update(1, 50)
    assert mapper.snapshot().frames_integrated == 2
    old = pipeline.product("mapping")
    pipeline.reset(2)
    assert mapper.snapshot().frames_integrated == 1
    assert mapper.snapshot().episode_id == 2 and old.episode_id == 1
    with pytest.raises(RuntimeError, match="Reset"):
        pipeline.update(3, 0)
    hidden = make_pipeline(PipelineConfig.localized(config), processors=(replace(service, expose_to_agent=False),))
    with pytest.raises(PermissionError, match="private"):
        hidden.product("mapping")


def test_service_dependencies_and_provenance_are_explicit():
    config = capture()
    available = ObservationAccess.from_config(config)
    nav = ServiceDeclaration("navigation", available, dependencies=("missing_map",))
    with pytest.raises(ValueError, match="dependencies"):
        ObservationPipeline.validate(PipelineConfig.localized(config), consumers=(nav,))
    with pytest.raises(ValueError, match="Duplicate"):
        declaration = ServiceDeclaration("navigation", available)
        ObservationPipeline.validate(PipelineConfig.localized(config), consumers=(declaration, declaration))


@dataclass(frozen=True)
class Integration:
    """What :class:`Integrator` has integrated this episode."""

    episode_id: int
    frames_integrated: int
    frame: str


class Integrator:
    """A processing service (as a camera mapper) counting the camera captures it integrates, in ``frame``."""

    def __init__(self, frame: str = "world"):
        self.frame = frame

    def reset(self, episode_id):
        self.episode_id, self.frames, self.stamps = episode_id, 0, set()

    def integrate(self, packet):
        stamps = {camera.stamp for camera in packet.cameras.values()}
        if stamps <= self.stamps:
            return False  # The same captures again.
        self.stamps |= stamps
        self.frames += 1
        return True

    def snapshot(self) -> Integration:
        return Integration(self.episode_id, self.frames, self.frame)


class Estimator:
    name = "test_estimator"
    inputs = ObservationAccess(imu=True)
    reference_frame = "local_map"

    def reset(self, episode_id):
        self.episode = episode_id
        self.calls = 0

    def update(self, packet):
        assert packet.localization is None and not packet.transforms and not packet.cameras
        self.calls += 1

    def localize(self, stamp):
        matrix = torch.eye(4)
        matrix[0, 3] = stamp.time_seconds
        return LocalizationObservation(
            stamp,
            Transform(self.reference_frame, "body", stamp, matrix, self.name),
            torch.tensor([1.0, 0, 0]),
            True,
            "estimated",
        )


def test_estimator_consumes_only_sensors_and_retains_capture_time_pose():
    config = capture("none")
    access = replace(ObservationAccess.from_config(config), localization="estimated")
    estimator = Estimator()
    pipeline = make_pipeline(PipelineConfig(config, access, access), localization=estimator)
    pipeline.update(1, 10)
    packet = pipeline.observe()
    assert packet.localization.mode == "estimated"
    assert packet.localization.pose.target == "local_map"
    assert packet.localization.pose.matrix[0, 3] == pytest.approx(0.04)  # The estimate at tick 10 (0.04 s).
    stamp = packet.cameras["front"].stamp
    assert stamp.tick == 0
    assert packet.transform("local_map", "front_optical", stamp)[0, 3] == pytest.approx(0.5)
    with pytest.raises(KeyError):
        packet.transform("world", "body", stamp)
    pipeline.update(1, 10)
    assert estimator.calls == 2
    pipeline.reset(2)
    assert estimator.calls == 1
    assert pipeline.observe().localization.stamp.episode_id == 2


def test_estimator_wrong_capture_time_fails_closed():
    class BadEstimator(Estimator):
        def localize(self, stamp):
            return super().localize(Stamp(stamp.episode_id + 1, stamp.tick, stamp.time_seconds))

    config = capture("none")
    access = replace(ObservationAccess.from_config(config), localization="estimated")
    with pytest.raises(ValueError, match="capture time"):
        make_pipeline(PipelineConfig(config, access, access), localization=BadEstimator())


class FakeNav2Transport:
    """Nav2's ROS side: a straight route and one command; nothing else."""

    def publish(self, state, raw_map):
        pass

    def start(self, goal):
        return ((0.0, 0.0), (goal.x, goal.y))

    def poll(self):
        return Nav2Update("active", (0.2, 0.0, 0.0))

    def cancel(self):
        pass

    def costmap(self):
        return None


def test_navigation_can_use_private_localization_while_agent_sees_only_camera():
    config = capture()
    maps = SimpleNamespace(configuration={"source": "observed_depth"}, snapshot=grid, raw_snapshot=grid)
    declared = Nav2BackendFactory(Nav2Config(), maps=maps)

    def factory(context):
        return Nav2NavigationBackend(
            context.runtime, context.locomotion, maps, context.config, Nav2Config(), FakeNav2Transport(),
            observations=context.observations, dead_band=SPOT_DEAD_BAND, coverage=SPOT_COVERAGE,
        )  # fmt: skip

    factory.declaration = declared.declaration
    agent = ObservationAccess(cameras=config.cameras)
    pipeline = make_pipeline(
        PipelineConfig(config, agent, ObservationAccess.from_config(config)),
        consumers=(factory.declaration,),
    )
    runtime = Runtime()
    runtime.observe = pipeline.observe
    client = NavigationClient(
        runtime,
        Motion(runtime),
        config=NavigationConfig(arm_posture=()),
        backend_factory=factory,
        observations=pipeline.view("navigation"),
    )
    command = client.navigate_to(1, 0)
    assert client.feedback(command).status == "active"
    assert client.feedback(command).path == ()
    assert client.backend.feedback(command).path
    assert runtime.observe().localization is None
    with pytest.raises(ValueError, match="goal frame"):
        client.navigate_to(1, 0, frame="private_scene_graph")


def test_complete_camera_stack_has_no_map_or_localization_dependency():
    received = []

    class CameraStack:
        declaration = ServiceDeclaration("navigation", ObservationAccess(cameras=(CameraPermission("front"),)))

        def __call__(self, context):
            assert not hasattr(context, "maps")
            packet = context.observations.observe()
            assert packet.localization is None
            assert "front" in packet.cameras
            return SimpleNamespace(
                configuration={"goal_frame": "native_map"},
                active=False,
                navigate_to=lambda x, y, heading, **kwargs: received.append((x, y, heading)) or 1,
            )

    config = PipelineConfig.observations_only(capture())
    factory = CameraStack()
    pipeline = make_pipeline(config, consumers=(factory.declaration,))
    runtime = Runtime()
    client = NavigationClient(
        runtime,
        Motion(runtime),
        backend_factory=factory,
        observations=pipeline.view("navigation"),
    )
    assert client.navigate_to(2, 3, frame="native_map") == 1
    assert received == [(2, 3, None)]


def test_body_relative_goal_is_resolved_in_declared_reference_at_submission():
    runtime = Runtime()
    runtime.x, runtime.y, runtime.yaw = 2.0, 3.0, 3.141592653589793 / 2
    received = []
    backend = SimpleNamespace(
        configuration={"goal_frame": "world"},
        navigate_to=lambda x, y, heading, **kwargs: received.append((x, y, heading)) or 1,
    )
    client = NavigationClient(runtime, Motion(runtime), backend_factory=lambda context: backend)
    client.navigate_to(1.0, 0.0, 0.0, frame="body")
    assert received == [pytest.approx((2.0, 4.0, 3.141592653589793 / 2))]


def test_pose_estimator_can_feed_mapping_in_its_own_reference_frame():
    config = capture("none")
    access = replace(ObservationAccess.from_config(config), localization="estimated")
    mapper = Integrator(frame="local_map")
    service = ProcessingService(
        ServiceDeclaration(
            "mapping",
            access,
            dependencies=("localization",),
            outputs=("occupancy_grid",),
        ),
        mapper,
    )
    pipeline = make_pipeline(PipelineConfig(config, access, access), processors=(service,), localization=Estimator())
    assert mapper.snapshot().frames_integrated == 1  # Integrated with the estimated pose.
    assert "test_estimator" in pipeline.configuration["services"][0]["effective_information_sources"]


def test_optional_scene_graph_retains_native_representation_and_detached_reads():
    class Graph:
        def reset(self, episode_id):
            self.graph = {"episode": episode_id, "nodes": []}

        def integrate(self, packet):
            assert packet.localization is None
            self.graph["nodes"] = [{"description": "agent-derived landmark"}]
            return True

        def snapshot(self):
            return self.graph

    config = PipelineConfig.observations_only(capture())
    graph = Graph()
    service = ProcessingService(
        ServiceDeclaration("scene_graph", config.agent, outputs=("semantic_graph",)),
        graph,
        expose_to_agent=True,
    )
    pipeline = make_pipeline(config, processors=(service,))
    snapshot = pipeline.product("scene_graph")
    snapshot["nodes"].clear()
    assert len(pipeline.product("scene_graph")["nodes"]) == 1
    assert pipeline.configuration["agent_products"] == ["scene_graph"]


def test_processor_failure_ends_episode_and_reset_does_not_reuse_state():
    from phygs.runtime.episode import EpisodeRuntime, EpisodeStatus, RunMode

    class FailingProcessor:
        def reset(self, episode_id):
            self.episode = episode_id

        def integrate(self, packet):
            if packet.stamp.tick:
                raise RuntimeError("processing failed")
            return True

        def snapshot(self):
            return self.episode

    config = PipelineConfig.observations_only(capture())
    processor = FailingProcessor()
    service = ProcessingService(ServiceDeclaration("model", config.agent), processor)
    pipeline = make_pipeline(config, processors=(service,))
    runtime = EpisodeRuntime(FakeWorld(("leg",)), (), mode=RunMode.AUTONOMOUS, observations=pipeline)
    runtime.reset()
    with pytest.raises(RuntimeError, match="processing failed"):
        runtime.step()
    assert runtime.status == EpisodeStatus.FAILED
    with pytest.raises(RuntimeError, match="failed"):
        runtime.step()
    runtime.reset()
    assert processor.episode == 2 and runtime.observe().stamp.tick == 0
