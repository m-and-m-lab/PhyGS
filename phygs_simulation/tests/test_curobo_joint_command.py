from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_legacy_curobo_joint_command_module_is_removed() -> None:
    manipulation_dir = _repo_root() / "scripts" / "skills" / "manipulation"

    assert not (manipulation_dir / "curobo_joint_command.py").exists()


def test_curobo_interface_uses_direct_esdf_import_not_local_pointcloud_esdf() -> None:
    source = (_repo_root() / "scripts" / "skills" / "manipulation" / "curobo_interfaces.py").read_text(encoding="utf-8")

    assert "update_voxel_data_from_external_esdf" in source
    assert "_build_point_cloud_occupancy" not in source
    assert "_build_esdf_query" not in source
