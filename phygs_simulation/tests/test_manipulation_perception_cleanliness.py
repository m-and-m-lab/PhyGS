from __future__ import annotations

from pathlib import Path


def test_manipulation_runtime_uses_generic_camera_utils_for_renderer_acquisition() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manipulation_dir = repo_root / "scripts" / "skills" / "manipulation"
    api_source = (manipulation_dir / "api.py").read_text(encoding="utf-8")
    runtime_source = (manipulation_dir / "manipulation_skill.py").read_text(encoding="utf-8")

    forbidden_snippets = (
        'cam_data.output.get("distance_to_image_plane")',
        'cam_data.output.get("rgb")',
        "crop_spatial_image",
        "crop_camera_intrinsics",
        "rgb_output",
    )
    for snippet in forbidden_snippets:
        assert snippet not in api_source
        assert snippet not in runtime_source
    assert "class SpotManipulationClient" in api_source
    assert "def grasp" in api_source
    assert "SpotManipulationRuntime" in api_source
    assert "AoGraspClient" in runtime_source
    assert "CuroboInterface" in runtime_source
    assert "CuroboTriggerRequest" in runtime_source


def test_manipulation_package_does_not_reexport_generic_image_utils() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    shim = repo_root / "scripts" / "skills" / "manipulation" / "image_utils.py"

    assert not shim.exists()
