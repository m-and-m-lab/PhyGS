from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import skills.manipulation.manipulation_skill as manipulation_skill
from helpers.cam_utils import DepthFilterConfig
from skills.manipulation import (
    AoGraspServiceConfig,
    CuroboInterfaceConfig,
    GraspPose,
    ManipulationState,
    SpotManipulationClient,
    SpotManipulationConfig,
    SpotManipulationGraspConfig,
    SpotManipulationGripperConfig,
    SpotManipulationPullConfig,
    SpotManipulationRobotConfig,
    SpotManipulationRobotHandles,
    SpotManipulationWorldConfig,
    resolve_spot_manipulation_robot_handles,
)


ARM_JOINT_NAMES = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")
GRIPPER_JOINT_NAMES = ("arm_f1x",)


class _FakeRobot:
    def __init__(self, *, missing_joint: str | None = None, missing_body: bool = False) -> None:
        names = (*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES, "leg_fl_hx")
        self.joint_names = tuple(name for name in names if name != missing_joint)
        self._joint_ids = {name: idx for idx, name in enumerate(self.joint_names)}
        self._body_ids = {} if missing_body else {"arm_link_fngr": 0}
        self.data = SimpleNamespace(
            joint_pos=torch.zeros((1, len(self.joint_names)), dtype=torch.float32),
            joint_vel=torch.zeros((1, len(self.joint_names)), dtype=torch.float32),
            root_pose_w=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pose_w=torch.tensor([[[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        )
        self.targets: list[tuple[tuple[int, ...], torch.Tensor]] = []

    def find_joints(self, names, preserve_order=True):
        ids = [self._joint_ids[name] for name in names if name in self._joint_ids]
        resolved = [name for name in names if name in self._joint_ids]
        return ids, resolved

    def find_bodies(self, names, preserve_order=True):
        ids = [self._body_ids[name] for name in names if name in self._body_ids]
        resolved = [name for name in names if name in self._body_ids]
        return ids, resolved

    def set_joint_position_target(self, target, joint_ids):
        ids = tuple(int(idx) for idx in joint_ids)
        target_tensor = torch.as_tensor(target, dtype=self.data.joint_pos.dtype)
        if target_tensor.ndim == 1:
            target_tensor = target_tensor.unsqueeze(0)
        self.targets.append((ids, target_tensor.clone()))
        self.data.joint_pos[:, list(ids)] = target_tensor


class _FakeScene:
    num_envs = 1


@dataclass
class _FakeAoGrasp:
    fail: bool = False

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def trigger(
        self,
        camera_map,
        camera_names,
        *,
        crops_by_camera,
        camera_prim_paths,
        depth_filter,
        debug,
    ):
        self.calls.append(
            {
                "camera_names": tuple(camera_names),
                "crops_by_camera": dict(crops_by_camera),
                "camera_prim_paths": dict(camera_prim_paths),
                "depth_filter": depth_filter,
                "debug": bool(debug),
            }
        )
        if self.fail:
            raise RuntimeError("AO-Grasp failed")
        return GraspPose(
            request_id="fake-grasp",
            position_world=np.array([0.5, 0.0, 0.4], dtype=np.float32),
            quaternion_world=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            position_fusion=np.array([0.1, 0.0, 0.2], dtype=np.float32),
            quaternion_fusion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            score=0.9,
            fusion_frame_name="hand",
        )


@dataclass
class _FakeCurobo:
    fail: bool = False

    def __post_init__(self) -> None:
        self.requests: list[object] = []

    def trigger(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("CuRobo failed")
        plan = SimpleNamespace(
            position=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float32),
            joint_names=ARM_JOINT_NAMES,
        )
        return SimpleNamespace(plan=plan)


class _FakeLocomotion:
    def __init__(self) -> None:
        self.velocity_commands: list[tuple[float, float, float]] = []
        self.step_count = 0
        self.stop_count = 0

    def command_velocity(self, x_mps: float, y_mps: float, yaw_rps: float) -> None:
        self.velocity_commands.append((float(x_mps), float(y_mps), float(yaw_rps)))

    def step(self) -> None:
        self.step_count += 1

    def stop(self) -> None:
        self.stop_count += 1


def _config(*, use_depth_collision: bool = True, close_steps: int = 0, pull_steps: int = 3) -> SpotManipulationConfig:
    robot = SpotManipulationRobotConfig(
        arm_joint_names=ARM_JOINT_NAMES,
        gripper_joint_names=GRIPPER_JOINT_NAMES,
        ee_body_name="arm_link_fngr",
    )
    return SpotManipulationConfig(
        robot=robot,
        camera_names=("hand",),
        depth_filter=DepthFilterConfig(max_depth_m=3.5),
        crops_by_camera={},
        camera_prim_paths={},
        ao_grasp=AoGraspServiceConfig(total_points=8),
        curobo=CuroboInterfaceConfig(
            robot_cfg="spot.yml",
            command_joint_names=robot.arm_joint_names,
            device="cpu",
            voxel_collision_enabled=use_depth_collision,
        ),
        gripper=SpotManipulationGripperConfig(
            open_rad=-1.0,
            closed_rad=-0.25,
            tolerance_rad=1e-4,
            open_timeout_steps=2,
            close_steps=close_steps,
            camera_settle_steps=0,
        ),
        grasp=SpotManipulationGraspConfig(pregrasp_offset_m=0.1, pregrasp_axis="z", pregrasp_sign=-1.0),
        pull=SpotManipulationPullConfig(backward_velocity_mps=0.25, steps=pull_steps),
        world=SpotManipulationWorldConfig(
            only_paths=("/World",),
            ignore_substring=("Robot",),
            robot_reference_prim_path="/World/envs/env_0/Robot",
        ),
    )


def _client(
    *,
    robot: _FakeRobot | None = None,
    ao_grasp: _FakeAoGrasp | None = None,
    curobo: _FakeCurobo | None = None,
    locomotion: _FakeLocomotion | None = None,
    cameras: dict[str, object] | None = None,
    config: SpotManipulationConfig | None = None,
) -> SpotManipulationClient:
    robot = robot or _FakeRobot()
    cfg = config or _config()
    handles = resolve_spot_manipulation_robot_handles(robot, cfg.robot)
    return SpotManipulationClient(
        robot=robot,
        scene=_FakeScene(),
        cameras={"hand": object()} if cameras is None else cameras,
        config=cfg,
        handles=handles,
        ao_grasp=ao_grasp or _FakeAoGrasp(),
        curobo=curobo or _FakeCurobo(),
        locomotion=locomotion,
        stage_getter=lambda: "stage",
    )


@pytest.fixture(autouse=True)
def _patch_isaaclab_transforms(monkeypatch):
    def combine_frame_transforms(t01, q01, t12=None, q12=None):
        return t01 if t12 is None else t01 + t12, q01 if q12 is None else q12

    def subtract_frame_transforms(t01, q01, t02=None, q02=None):
        return t02 if t02 is not None else -t01, q02 if q02 is not None else q01

    monkeypatch.setattr(manipulation_skill, "_combine_frame_transforms", combine_frame_transforms)
    monkeypatch.setattr(manipulation_skill, "_subtract_frame_transforms", subtract_frame_transforms)


def _run_until_terminal(client: SpotManipulationClient, *, max_steps: int = 32) -> list[ManipulationState]:
    states: list[ManipulationState] = []
    for _ in range(max_steps):
        feedback = client.step()
        states.append(feedback.state)
        if feedback.is_terminal:
            return states
    raise AssertionError(f"Manipulation command did not terminate. Last state={client.state}")


def test_resolves_robot_handles_strictly() -> None:
    robot = _FakeRobot()
    handles = resolve_spot_manipulation_robot_handles(robot, _config().robot)

    assert handles == SpotManipulationRobotHandles(
        arm_joint_ids=(0, 1, 2, 3, 4, 5),
        gripper_joint_ids=(6,),
        ee_body_id=0,
    )


def test_resolve_handles_fails_on_missing_joint_or_body() -> None:
    with pytest.raises(RuntimeError, match="arm joints"):
        resolve_spot_manipulation_robot_handles(_FakeRobot(missing_joint="arm_wr1"), _config().robot)

    with pytest.raises(RuntimeError, match="EE body"):
        resolve_spot_manipulation_robot_handles(_FakeRobot(missing_body=True), _config().robot)


def test_grasp_runs_aograsp_then_two_curobo_plans_and_closes_gripper() -> None:
    ao_grasp = _FakeAoGrasp()
    curobo = _FakeCurobo()
    client = _client(ao_grasp=ao_grasp, curobo=curobo)

    command_id = client.grasp(debug=True)
    states = _run_until_terminal(client)

    assert command_id == 1
    assert states[-1] == ManipulationState.DONE
    assert ManipulationState.PERCEIVING in states
    assert ManipulationState.EXECUTING_PREGRASP in states
    assert ManipulationState.EXECUTING_GRASP in states
    assert ao_grasp.calls[0]["camera_names"] == ("hand",)
    assert ao_grasp.calls[0]["debug"] is True
    assert len(curobo.requests) == 2
    assert torch.allclose(client.robot.data.joint_pos[:, [6]], torch.tensor([[-0.25]]))


def test_release_opens_gripper_without_perception_or_planning() -> None:
    ao_grasp = _FakeAoGrasp()
    curobo = _FakeCurobo()
    client = _client(ao_grasp=ao_grasp, curobo=curobo)
    client.robot.data.joint_pos[:, [6]] = torch.tensor([[-0.25]])

    command_id = client.release()
    states = _run_until_terminal(client)

    assert command_id == 1
    assert states[-1] == ManipulationState.DONE
    assert ao_grasp.calls == []
    assert curobo.requests == []
    assert torch.allclose(client.robot.data.joint_pos[:, [6]], torch.tensor([[-1.0]]))


def test_interact_articulated_grasps_pulls_backward_and_releases() -> None:
    locomotion = _FakeLocomotion()
    client = _client(locomotion=locomotion, config=_config(pull_steps=3))

    client.interact_articulated()
    states = _run_until_terminal(client)

    assert states[-1] == ManipulationState.DONE
    assert ManipulationState.PULLING_BACK in states
    assert locomotion.velocity_commands == [(-0.25, 0.0, 0.0)] * 3
    assert locomotion.stop_count == 1
    assert locomotion.step_count == 4
    assert torch.allclose(client.robot.data.joint_pos[:, [6]], torch.tensor([[-1.0]]))


def test_interact_articulated_requires_locomotion() -> None:
    client = _client()

    with pytest.raises(RuntimeError, match="locomotion"):
        client.interact_articulated()


def test_constructor_fails_on_missing_camera() -> None:
    with pytest.raises(RuntimeError, match="Missing configured manipulation cameras"):
        _client(cameras={})


def test_aograsp_failure_marks_command_failed_and_raises() -> None:
    client = _client(ao_grasp=_FakeAoGrasp(fail=True))
    client.grasp()

    with pytest.raises(RuntimeError, match="AO-Grasp failed"):
        for _ in range(8):
            client.step()

    assert client.state == ManipulationState.FAILED
    assert "AO-Grasp failed" in client.last_error


def test_curobo_failure_marks_command_failed_and_raises() -> None:
    client = _client(curobo=_FakeCurobo(fail=True))
    client.grasp()

    with pytest.raises(RuntimeError, match="CuRobo failed"):
        for _ in range(8):
            client.step()

    assert client.state == ManipulationState.FAILED
    assert "CuRobo failed" in client.last_error


def test_depth_collision_planning_does_not_pass_stage_to_curobo_trigger() -> None:
    curobo = _FakeCurobo()
    client = _client(curobo=curobo)

    client.grasp()
    _run_until_terminal(client)

    assert len(curobo.requests) == 2
    assert all(request.stage is None for request in curobo.requests)
    assert all(request.stage_reference_prim_path is None for request in curobo.requests)


def test_usd_collision_planning_passes_stage_scope_to_curobo_trigger() -> None:
    curobo = _FakeCurobo()
    client = _client(curobo=curobo, config=_config(use_depth_collision=False))

    client.grasp()
    _run_until_terminal(client)

    assert len(curobo.requests) == 2
    assert all(request.stage == "stage" for request in curobo.requests)
    assert all(request.stage_only_paths == ("/World",) for request in curobo.requests)
    assert all(request.stage_ignore_substring == ("Robot",) for request in curobo.requests)
    assert all(request.stage_reference_prim_path == "/World/envs/env_0/Robot" for request in curobo.requests)
