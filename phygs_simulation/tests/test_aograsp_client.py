from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.cam_utils import DepthFilterConfig
from skills.manipulation.aograsp_client import (
    AoGraspClient,
    AoGraspServiceConfig,
    AoGraspServiceError,
)


class _FakeAoGraspClient(AoGraspClient):
    def __init__(
        self,
        *,
        total_points: int = 4,
        heatmap=None,
        proposals=None,
        health=None,
        debug_root: str = "outputs/test-aograsp",
        **config_overrides,
    ) -> None:
        config_overrides.setdefault("proposal_viz_output_root", None)
        super().__init__(
            AoGraspServiceConfig(
                pointscore_url="http://pointscore",
                cgn_url="http://cgn",
                debug_root=debug_root,
                total_points=total_points,
                rng_seed=0,
                **config_overrides,
            )
        )
        self.calls: list[tuple[str, str, dict | None]] = []
        self._heatmap = heatmap
        self._proposals = proposals
        self._health = health or {
            "pointscore": {"status": "ok", "expected_points": total_points},
            "cgn": {"status": "ok", "expected_points": total_points},
        }

    def _request_json(self, method: str, url: str, payload=None):
        self.calls.append((method, url, payload))
        if url == "http://pointscore/healthz":
            return self._health["pointscore"]
        if url == "http://cgn/healthz":
            return self._health["cgn"]
        if url == "http://pointscore/heatmaps":
            points = np.asarray(payload["points_cam"], dtype=np.float32)
            labels = self._heatmap
            if labels is None:
                labels = np.linspace(0.0, 1.0, num=points.shape[0], dtype=np.float32)
            return {"request_id": payload["request_id"], "labels": np.asarray(labels, dtype=np.float32).tolist()}
        if url == "http://cgn/proposals":
            proposals = self._proposals
            if proposals is None:
                proposals = [
                    {
                        "position_cam": [0.0, 0.0, 1.0],
                        "quaternion_cam": [1.0, 0.0, 0.0, 0.0],
                        "score": 0.9,
                    }
                ]
            return {"request_id": payload["request_id"], "proposals": proposals}
        raise AssertionError(f"Unexpected request: {method} {url}")


def _fake_camera(*, position_w=(0.0, 0.0, 0.0), depth=None):
    if depth is None:
        depth = np.ones((1, 2, 2, 1), dtype=np.float32)
    rgb = np.full((1, 2, 2, 3), 40, dtype=np.uint8)
    return SimpleNamespace(
        data=SimpleNamespace(
            output={
                "distance_to_image_plane": depth,
                "rgb": rgb,
            },
            intrinsic_matrices=np.array(
                [[[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
            pos_w=np.asarray([position_w], dtype=np.float32),
            quat_w_ros=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
    )


def _camera_map():
    return {
        "front": _fake_camera(position_w=(1.0, 0.0, 0.0)),
        "hand": _fake_camera(position_w=(1.1, 0.0, 0.0)),
    }


def _require_open3d_offscreen_window() -> None:
    o3d = pytest.importorskip("open3d")
    vis = o3d.visualization.Visualizer()
    created = vis.create_window(width=16, height=16, visible=False)
    if created:
        vis.destroy_window()
    if not created:
        pytest.skip("Open3D cannot create an offscreen window in this environment.")


def test_trigger_returns_best_world_pose() -> None:
    client = _FakeAoGraspClient(total_points=4)

    pose = client.trigger(
        _camera_map(),
        ("front", "hand"),
        depth_filter=DepthFilterConfig(max_depth_m=3.5),
    )

    assert pose.request_id
    assert pose.fusion_frame_name == "front"
    assert pose.score == pytest.approx(0.9)
    assert np.allclose(pose.position_fusion, [0.0, 0.0, 1.0])
    assert np.allclose(pose.position_world, [1.0, 0.0, 1.0])
    assert [call[1] for call in client.calls] == [
        "http://pointscore/healthz",
        "http://cgn/healthz",
        "http://pointscore/heatmaps",
        "http://cgn/proposals",
    ]


def test_trigger_rejects_health_point_count_mismatch() -> None:
    client = _FakeAoGraspClient(
        total_points=4,
        health={
            "pointscore": {"status": "ok", "expected_points": 8},
            "cgn": {"status": "ok", "expected_points": 4},
        },
    )

    with pytest.raises(AoGraspServiceError, match="point-count contract mismatch"):
        client.trigger(_camera_map(), ("front", "hand"))


def test_trigger_rejects_invalid_heatmap_shape() -> None:
    client = _FakeAoGraspClient(total_points=4, heatmap=[0.1, 0.2, 0.3])

    with pytest.raises(AoGraspServiceError, match="labels shape"):
        client.trigger(_camera_map(), ("front", "hand"))


def test_trigger_rejects_empty_proposals() -> None:
    client = _FakeAoGraspClient(total_points=4, proposals=[])

    with pytest.raises(AoGraspServiceError, match="did not return any grasp proposals"):
        client.trigger(_camera_map(), ("front", "hand"))


def test_trigger_writes_debug_artifacts(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    client = _FakeAoGraspClient(total_points=4, debug_root=str(tmp_path))

    pose = client.trigger(_camera_map(), ("front", "hand"), debug=True)

    output_dir = tmp_path / pose.request_id
    assert (output_dir / "points_fusion.npy").exists()
    assert (output_dir / "points_world.npy").exists()
    assert (output_dir / "heatmap.npy").exists()
    assert (output_dir / "proposal.json").exists()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["request_id"] == pose.request_id
    assert metadata["fusion_frame_name"] == "front"
    assert metadata["points_fusion_shape"] == [4, 3]
    assert set(metadata["depth_preview_files"]) == {"front", "hand"}
    assert set(metadata["rgb_image_files"]) == {"front", "hand"}
    assert metadata["heatmap_shape"] == [4]
    assert metadata["proposal_file"] == "proposal.json"


def test_trigger_writes_local_aograsp_visualizations(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    _require_open3d_offscreen_window()
    output_root = tmp_path / "output"
    client = _FakeAoGraspClient(
        total_points=4,
        debug_root=str(tmp_path / "debug"),
        proposal_viz_output_root=str(output_root),
        proposal_viz_top_k=10,
    )

    pose = client.trigger(_camera_map(), ("front", "hand"), debug=True)

    run_output_root = output_root / pose.request_id
    heatmap_file = run_output_root / "point_score" / f"{pose.request_id}.npz"
    heatmap_image_file = run_output_root / "point_score_img" / f"heatmap_{pose.request_id}.png"
    heatmap_histogram_file = run_output_root / "point_score_img" / f"heatmap_{pose.request_id}_hist.png"
    proposal_file = run_output_root / "grasp_proposals" / f"{pose.request_id}.npz"
    proposal_image_file = run_output_root / "grasp_proposals_img" / f"{pose.request_id}.png"
    metadata = json.loads((tmp_path / "debug" / pose.request_id / "metadata.json").read_text(encoding="utf-8"))
    assert heatmap_file.exists()
    assert heatmap_image_file.exists()
    assert heatmap_histogram_file.exists()
    assert proposal_file.exists()
    assert proposal_image_file.exists()
    assert metadata["aograsp_heatmap_file"] == str(heatmap_file)
    assert metadata["aograsp_heatmap_image_file"] == str(heatmap_image_file)
    assert metadata["aograsp_heatmap_histogram_file"] == str(heatmap_histogram_file)
    assert metadata["aograsp_grasp_proposals_file"] == str(proposal_file)
    assert metadata["aograsp_grasp_proposals_image_file"] == str(proposal_image_file)
