from __future__ import annotations

from pathlib import Path

import pytest

from skills.locomotion import build_policy_to_runtime_indices, load_spot_arm_policy_config, parse_policy_joint_order


def test_parse_policy_joint_order_reads_robot_joint_names_from_usda() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    asset_path = repo_root / "spot_model" / "spot_arm_w_cam.usda"

    assert parse_policy_joint_order(asset_path) == (
        "arm_sh0",
        "arm_sh1",
        "arm_el0",
        "arm_el1",
        "arm_wr0",
        "arm_wr1",
        "arm_f1x",
        "front_left_hip_x",
        "front_left_hip_y",
        "front_left_knee",
        "front_right_hip_x",
        "front_right_hip_y",
        "front_right_knee",
        "rear_left_hip_x",
        "rear_left_hip_y",
        "rear_left_knee",
        "rear_right_hip_x",
        "rear_right_hip_y",
        "rear_right_knee",
    )


def test_build_policy_to_runtime_indices_uses_exact_joint_names() -> None:
    policy_joint_names = ("arm_sh0", "front_left_hip_y", "rear_right_knee")
    runtime_joint_names = ("rear_right_knee", "arm_sh0", "front_left_hip_y")

    assert build_policy_to_runtime_indices(policy_joint_names, runtime_joint_names) == (1, 2, 0)


def test_build_policy_to_runtime_indices_fails_fast_on_missing_joint() -> None:
    with pytest.raises(ValueError, match="missing policy joint 'front_left_hip_y'"):
        build_policy_to_runtime_indices(("front_left_hip_y",), ("fl_hy",))


def test_load_spot_arm_policy_config_resolves_leg_only_joint_subset_from_env_yaml() -> None:
    workspace_root = Path(__file__).resolve().parents[3]
    asset_path = workspace_root / "scripts" / "phygs_simulation" / "spot_model" / "spot_arm_w_cam.usda"
    env_config_path = (
        workspace_root / "logs" / "rsl_rl" / "spot_arm_w_cam_flat" / "2026-04-23_14-29-35" / "params" / "env.yaml"
    )
    policy_path = (
        workspace_root / "logs" / "rsl_rl" / "spot_arm_w_cam_flat" / "2026-04-23_14-29-35" / "exported" / "policy.pt"
    )

    config = load_spot_arm_policy_config(
        policy_path=str(policy_path),
        env_config_path=str(env_config_path),
        asset_path=str(asset_path),
    )

    assert config.policy_joint_names == (
        "front_left_hip_x",
        "front_left_hip_y",
        "front_left_knee",
        "front_right_hip_x",
        "front_right_hip_y",
        "front_right_knee",
        "rear_left_hip_x",
        "rear_left_hip_y",
        "rear_left_knee",
        "rear_right_hip_x",
        "rear_right_hip_y",
        "rear_right_knee",
    )
    assert config.action_dim == 12
    assert config.observation_dim == 48
    assert config.render_interval == 10
    assert config.gravity == (0.0, 0.0, -9.81)
    assert config.asset_joint_stiffness[:6] == (400.0, 400.0, 400.0, 400.0, 400.0, 400.0)
    assert config.asset_joint_damping[:6] == (60.0, 60.0, 60.0, 60.0, 60.0, 60.0)
