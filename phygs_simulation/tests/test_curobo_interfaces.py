from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None
    pytestmark = pytest.mark.skip(reason="torch is required for curobo interface tests")

if torch is not None:
    from helpers.cam_utils import DepthFilterConfig
    from skills.manipulation import curobo_interfaces
    from skills.manipulation.curobo_interfaces import (
        CuroboInputError,
        CuroboInterface,
        CuroboInterfaceConfig,
        CuroboPlanningError,
        CuroboTriggerRequest,
    )


class _FakeTensorArgs:
    device = torch.device("cpu") if torch is not None else None
    dtype = torch.float32 if torch is not None else None

    def to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(value), device=self.device, dtype=self.dtype)


class _FakePose:
    def __init__(self, *, position, quaternion) -> None:
        self.position = position
        self.quaternion = quaternion


class _FakeJointState:
    def __init__(self, *, position, velocity=None, acceleration=None, jerk=None, joint_names) -> None:
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        self.joint_names = list(joint_names)

    def get_ordered_joint_state(self, joint_names):
        indices = [self.joint_names.index(name) for name in joint_names]

        def select(value):
            if value is None:
                return None
            if value.ndim == 1:
                return value[indices]
            return value[:, indices]

        return _FakeJointState(
            position=select(self.position),
            velocity=select(self.velocity),
            acceleration=select(self.acceleration),
            jerk=select(self.jerk),
            joint_names=list(joint_names),
        )

    def unsqueeze(self, _dim):
        return self


class _FakeResult:
    def __init__(self, *, success: bool, status: str, plan=None) -> None:
        self.success = torch.tensor(bool(success))
        self.status = status
        self._plan = plan

    def get_interpolated_plan(self):
        return self._plan


class _FakeMotionGen:
    def __init__(self, *, success: bool = True) -> None:
        self.kinematics = SimpleNamespace(
            joint_names=["j1", "j2"],
            get_joint_limits=lambda: SimpleNamespace(
                position=torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32),
                joint_names=["j1", "j2"],
            ),
        )
        self.success = success
        self.calls = []
        self.world_collision = SimpleNamespace(update_voxel_data_from_external_esdf=lambda *args, **kwargs: None)

    def plan_single(self, start_state, goal_pose, plan_config):
        self.calls.append((start_state, goal_pose, plan_config))
        if not self.success:
            return _FakeResult(success=False, status="IK_FAIL")
        plan = _FakeJointState(
            position=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            velocity=torch.tensor([[0.0, 0.0], [0.1, 0.1]], dtype=torch.float32),
            joint_names=["j1", "j2"],
        )
        return _FakeResult(success=True, status="OK", plan=plan)

    def get_full_js(self, plan):
        return plan


def _runtime(motion_gen: _FakeMotionGen):
    return curobo_interfaces._CuroboRuntime(
        motion_gen=motion_gen,
        tensor_args=_FakeTensorArgs(),
        plan_config=object(),
        pose_cls=_FakePose,
        joint_state_cls=_FakeJointState,
    )


def _fake_camera(*, position_w=(0.0, 0.0, 0.0), depth=None):
    if depth is None:
        depth = np.ones((1, 2, 2, 1), dtype=np.float32)
    return SimpleNamespace(
        data=SimpleNamespace(
            output={
                "distance_to_image_plane": depth,
                "rgb": np.full((1, 2, 2, 3), 55, dtype=np.uint8),
            },
            intrinsic_matrices=np.array(
                [[[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
            pos_w=np.asarray([position_w], dtype=np.float32),
            quat_w_ros=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
    )


def _request(**overrides) -> CuroboTriggerRequest:
    values = {
        "camera_map": {"front": _fake_camera()},
        "camera_names": ("front",),
        "joint_positions": np.array([0.0, 0.0], dtype=np.float32),
        "joint_velocities": np.array([0.0, 0.0], dtype=np.float32),
        "joint_names": ("j1", "j2"),
        "target_position_w": np.array([0.4, 0.0, 0.3], dtype=np.float32),
        "target_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "depth_filter": DepthFilterConfig(max_depth_m=3.5),
        "stage": "stage",
        "stage_reference_prim_path": "/World/Robot",
    }
    values.update(overrides)
    return CuroboTriggerRequest(**values)


def _interface(motion_gen: _FakeMotionGen, tmp_path) -> CuroboInterface:
    return CuroboInterface(
        CuroboInterfaceConfig(
            robot_cfg={},
            command_joint_names=("j2", "j1"),
            debug_root=str(tmp_path),
            warmup=False,
            voxel_collision_enabled=False,
        ),
        runtime=_runtime(motion_gen),
    )


@pytest.fixture(autouse=True)
def _patch_stage_update(monkeypatch):
    calls = []

    def update_stage(self, stage, *, only_paths, ignore_substring, reference_prim_path):
        calls.append(
            {
                "stage": stage,
                "only_paths": tuple(only_paths),
                "ignore_substring": tuple(ignore_substring),
                "reference_prim_path": reference_prim_path,
            }
        )

    monkeypatch.setattr(CuroboInterface, "_update_world_from_stage", update_stage)
    return calls


def test_trigger_runs_camera_pipeline_and_returns_ordered_plan(tmp_path) -> None:
    motion_gen = _FakeMotionGen()
    interface = _interface(motion_gen, tmp_path)

    result = interface.trigger(
        _request(
            include_rgb=False,
            include_pointcloud=False,
            request_id="request_a",
        )
    )

    assert result.request_id == "request_a"
    assert result.status == "OK"
    assert len(result.camera_pipeline.frames) == 1
    assert result.camera_pipeline.frames[0].rgb_image is None
    assert result.camera_pipeline.fused_pointcloud is None
    assert result.plan.joint_names == ["j2", "j1"]
    assert np.allclose(result.plan.position, [[0.2, 0.1], [0.4, 0.3]])
    assert len(motion_gen.calls) == 1
    assert motion_gen.calls[0][1].position.tolist() == pytest.approx([0.4, 0.0, 0.3])


def test_trigger_raises_when_planning_fails(tmp_path) -> None:
    interface = _interface(_FakeMotionGen(success=False), tmp_path)

    with pytest.raises(CuroboPlanningError, match="IK_FAIL"):
        interface.trigger(_request())


def test_trigger_raises_on_joint_limit_violation(tmp_path) -> None:
    interface = _interface(_FakeMotionGen(), tmp_path)

    with pytest.raises(CuroboInputError, match="violates joint limits"):
        interface.trigger(_request(joint_positions=np.array([2.0, 0.0], dtype=np.float32)))


def test_trigger_writes_debug_artifacts(tmp_path) -> None:
    interface = _interface(_FakeMotionGen(), tmp_path)

    result = interface.trigger(
        _request(
            include_pointcloud=True,
            total_points=4,
            request_id="debug_request",
            debug=True,
        )
    )

    output_dir = tmp_path / "debug_request"
    assert result.debug_dir == output_dir
    assert (output_dir / "front_depth.npy").exists()
    assert (output_dir / "front_rgb.npy").exists()
    assert (output_dir / "points_fusion.npy").exists()
    assert (output_dir / "points_world.npy").exists()
    assert (output_dir / "plan_position.npy").exists()
    assert (output_dir / "plan_velocity.npy").exists()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["request_id"] == "debug_request"
    assert metadata["plan_joint_names"] == ["j2", "j1"]
    assert metadata["plan_position_shape"] == [2, 2]
    assert metadata["cloud"]["points_fusion_shape"] == [4, 3]


def test_usd_mode_updates_stage_and_does_not_update_esdf(tmp_path, monkeypatch, _patch_stage_update) -> None:
    interface = _interface(_FakeMotionGen(), tmp_path)
    esdf_calls = []
    monkeypatch.setattr(interface, "_update_voxel_world_from_frames", lambda frames: esdf_calls.append(frames))

    interface.trigger(
        _request(
            stage="stage_a",
            stage_only_paths=("/World",),
            stage_ignore_substring=("Robot",),
            stage_reference_prim_path="/World/Robot",
        )
    )

    assert _patch_stage_update == [
        {
            "stage": "stage_a",
            "only_paths": ("/World",),
            "ignore_substring": ("Robot",),
            "reference_prim_path": "/World/Robot",
        }
    ]
    assert esdf_calls == []


def test_depth_mode_updates_esdf_and_does_not_update_stage(tmp_path, monkeypatch, _patch_stage_update) -> None:
    interface = _interface(_FakeMotionGen(), tmp_path)
    interface.config = CuroboInterfaceConfig(
        robot_cfg={},
        command_joint_names=("j2", "j1"),
        debug_root=str(tmp_path),
        warmup=False,
        voxel_collision_enabled=True,
        voxel_bounds_min=(0.0, 0.0, 0.0),
        voxel_bounds_max=(1.0, 1.0, 1.0),
    )
    esdf_calls = []
    monkeypatch.setattr(interface, "_update_voxel_world_from_frames", lambda frames: esdf_calls.append(tuple(frames)) or {"grid": np.zeros((1, 1, 1), dtype=np.float32), "dims": [1, 1, 1], "voxel_size": 1.0, "origin": [0.0, 0.0, 0.0]})

    interface.trigger(_request(stage=None, stage_reference_prim_path=None))

    assert len(esdf_calls) == 1
    assert _patch_stage_update == []


def test_usd_mode_requires_stage_inputs(tmp_path) -> None:
    interface = _interface(_FakeMotionGen(), tmp_path)

    with pytest.raises(CuroboInputError, match="stage is required"):
        interface.trigger(_request(stage=None))

    with pytest.raises(CuroboInputError, match="stage_reference_prim_path"):
        interface.trigger(_request(stage="stage", stage_reference_prim_path=None))
