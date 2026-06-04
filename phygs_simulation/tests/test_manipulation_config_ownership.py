from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from skills.manipulation import load_spot_manipulation_config
from helpers.cam_utils import ImageCrop, load_camera_rig_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_drawer_smoke_yaml_only_owns_scene_and_test_inputs() -> None:
    config_path = _repo_root() / "tests" / "isaaclab_test" / "spot_manipulation_drawer.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "manipulation_profile" not in config
    assert "cameras" not in config
    assert "ao_grasp" not in config
    assert "curobo" not in config
    assert "camera_warmup_steps" not in config.get("timeouts", {})

    robot_config = config["robot"]
    forbidden_robot_keys = {
        "usd_path",
        "arm_joint_names",
        "gripper_joint_names",
        "ee_body_name",
        "disable_gravity",
        "fix_root_link",
    }
    assert forbidden_robot_keys.isdisjoint(robot_config)


def test_camera_rig_config_owns_camera_paths_crops_and_depth_filters() -> None:
    rig_path = _repo_root() / "scripts" / "helpers" / "config" / "spot_arm_cameras.yaml"
    rig = load_camera_rig_config(rig_path)

    assert rig.camera_names == ("frontleft", "frontright", "hand")
    assert rig.grasp_camera_names == ("frontleft", "frontright", "hand")
    assert rig.depth_collision_camera_names == ("frontleft", "frontright", "hand")
    assert rig.cameras_by_name["hand"].crop == ImageCrop(left=38)
    assert rig.cameras_by_name["frontleft"].depth_filter.max_depth_m == 3.5


def test_spot_manipulation_config_composes_owned_configs(monkeypatch) -> None:
    monkeypatch.setenv("AO_POINTSCORE_URL", "http://127.0.0.1:18081")
    monkeypatch.setenv("AO_CGN_URL", "http://127.0.0.1:18082")
    config = load_spot_manipulation_config(device="cpu")

    assert config.robot.arm_joint_names == ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")
    assert config.robot.gripper_joint_names == ("arm_f1x",)
    assert config.robot.ee_body_name == "arm_link_fngr"
    assert config.camera_names == ("frontleft", "frontright", "hand")
    assert config.ao_grasp.pointscore_url == "http://127.0.0.1:18081"
    assert config.ao_grasp.cgn_url == "http://127.0.0.1:18082"
    assert config.ao_grasp.total_points == 16384
    assert "Gemini2_front_left" in config.camera_prim_paths["frontleft"]
    assert config.camera_prim_paths["frontleft"].endswith("/Stream_depth")
    assert config.ao_grasp.reference_prim_path == "{ENV_REGEX_NS}/Robot/body"
    assert config.ao_grasp.fusion_frame is not None
    assert config.ao_grasp.fusion_frame.mode == "body_aligned_stereo_midpoint"
    assert config.ao_grasp.fusion_frame.name == "front_center"
    assert config.ao_grasp.fusion_frame.left_camera_name == "frontleft"
    assert config.ao_grasp.fusion_frame.right_camera_name == "frontright"
    assert config.ao_grasp.debug_root == "/workspace/isaaclab/outputs/ao-grasp"
    assert config.ao_grasp.pointcloud_filter.min_z_m == pytest.approx(0.03)
    assert config.ao_grasp.pointcloud_filter.bounds_min is None
    assert config.ao_grasp.pointcloud_filter.bounds_max is None
    assert config.ao_grasp.pointcloud_filter.voxel_size_m == pytest.approx(0.01)
    assert config.ao_grasp.proposal_viz_output_root == "/workspace/isaaclab/outputs/ao-grasp"
    assert config.ao_grasp.proposal_viz_top_k == 10
    assert config.curobo.voxel_collision_enabled is True
    assert config.curobo.voxel_bounds_min == pytest.approx((0.0, -0.75, 0.0))
    assert config.curobo.voxel_bounds_max == pytest.approx((1.35, 0.75, 1.20))
    assert config.curobo.voxel_size_m == pytest.approx(0.025)
    assert config.curobo.truncation_distance_vox == pytest.approx(2.0)
    assert config.curobo.raycast_subsampling == 1
    assert config.curobo.accumulate_depth_frames is False
    assert config.curobo.import_invert_sign is True
    assert config.curobo.import_add_half_voxel is True
    curobo_runtime_path = _repo_root() / "scripts" / "skills" / "manipulation" / "config" / "curobo_spot_runtime.yaml"
    runtime_config = yaml.safe_load(curobo_runtime_path.read_text(encoding="utf-8"))
    assert "update_world" not in runtime_config


def test_generic_camera_utils_do_not_import_skills_or_planners() -> None:
    source = (_repo_root() / "scripts" / "helpers" / "cam_utils.py").read_text(encoding="utf-8")

    assert "skills." not in source
    assert "aograsp" not in source.lower()
    assert "curobo" not in source.lower()


def test_generic_pointcloud_utils_do_not_encode_algorithm_contracts() -> None:
    source = (_repo_root() / "scripts" / "helpers" / "pcd_utils.py").read_text(encoding="utf-8")

    assert "aograsp" not in source.lower()
    assert "curobo" not in source.lower()


def test_manipulation_runtime_values_are_not_hidden_in_defaults_module() -> None:
    repo_root = _repo_root()

    assert not (repo_root / "scripts" / "skills" / "manipulation" / "defaults.py").exists()


def test_manipulation_types_are_not_hidden_in_a_separate_bucket_module() -> None:
    repo_root = _repo_root()

    assert not (repo_root / "scripts" / "skills" / "manipulation" / "grasp_types.py").exists()


def test_public_api_stays_separate_from_runtime_state_machine() -> None:
    repo_root = _repo_root()
    manipulation_dir = repo_root / "scripts" / "skills" / "manipulation"
    api_source = (manipulation_dir / "api.py").read_text(encoding="utf-8")

    assert (manipulation_dir / "manipulation_skill.py").exists()
    assert "class SpotManipulationClient" in api_source
    assert "def grasp" in api_source
    assert "def step" in api_source
    assert "class _PlanCursor" not in api_source
    assert "def _combine_frame_transforms" not in api_source
    assert not (manipulation_dir / "isaaclab_spot_backend.py").exists()
    assert not (manipulation_dir / "runtime_config.py").exists()
