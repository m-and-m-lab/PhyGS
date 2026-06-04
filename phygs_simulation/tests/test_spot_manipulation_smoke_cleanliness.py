from __future__ import annotations

from pathlib import Path


def test_drawer_smoke_test_does_not_orchestrate_gripper_or_target_center() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tests" / "isaaclab_test" / "spot_manipulation_drawer.py"
    source = script.read_text(encoding="utf-8")

    forbidden_snippets = (
        "SPOT_GRIPPER_OPEN_RAD",
        "SPOT_GRIPPER_TOLERANCE_RAD",
        "RobotCommandBuilder",
        "_open_gripper_for_capture",
        "def _open_gripper",
        "def _wait_for_gripper_open",
        "compute_target_world_bounds",
        "target_center_b=",
        "subtract_frame_transforms",
        "gripper_open_timeout_sec",
        "gripper_settle_steps",
        "_require_aograsp_healthcheck",
        "_require_aograsp_point_contract",
        "_aograsp_connection_hint",
        "_require_ready_robot_state",
        "_require_ready_cameras",
        "class CameraConfig",
        "class AoGraspConfig",
        "class CuroboConfig",
        "self._config.ao_grasp",
        "self._config.curobo",
        "self._config.cameras",
        "manipulation_profile:",
        "root.manipulation_profile",
        "_sanitize_camera_path",
        "_resolve_camera_map",
        "camera_names=",
        "camera_warmup_steps",
        "renderer_cameras",
        "self._profile.camera_rig",
    )
    for snippet in forbidden_snippets:
        assert snippet not in source
