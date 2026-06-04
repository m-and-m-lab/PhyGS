#!/usr/bin/env python3
"""Open an interactive Open3D viewer for saved AO-Grasp debug artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from helpers.aograsp_viz import show_aograsp_visualization


DEFAULT_ROOT = Path("/workspace/isaaclab/outputs/ao-grasp")


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run, root=args.root)
    points, heatmap, proposals = _load_run(run_dir)
    body_frame = _load_body_frame(run_dir)

    print(f"AO-Grasp run: {run_dir}")
    print("Fusion-frame axes at origin: X red, Y green, Z blue.")
    if body_frame is not None:
        print("Body/reference axes in fusion view: X magenta, Y cyan, Z yellow.")
        print("Expected mapping: +Z_fusion=+X_body, +X_fusion=-Y_body, +Y_fusion=-Z_body.")
    print("Point colors: red/yellow/green are low/mid/high AO-Grasp heatmap scores.")
    print(f"Showing top {min(args.top_k, len(proposals))} proposal(s). Close the Open3D window to exit.")

    show_aograsp_visualization(
        points_fusion=points,
        heatmap=heatmap,
        proposals=proposals,
        top_k=args.top_k,
        body_frame=body_frame,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="AO-Grasp run tag, run directory, or proposal/point_score .npz path.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"AO-Grasp output root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top proposals to draw.")
    return parser.parse_args()


def _resolve_run_dir(value: str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.suffix == ".npz":
        return path.resolve().parents[1]
    if path.exists():
        return path.resolve()
    run_dir = (root.expanduser().resolve() / value).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"AO-Grasp run was not found: {run_dir}")
    return run_dir


def _load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, object]]]:
    tag = run_dir.name
    proposal_file = run_dir / "grasp_proposals" / f"{tag}.npz"
    if proposal_file.exists():
        data = np.load(proposal_file, allow_pickle=True)["data"].item()
        return (
            np.asarray(data["input"], dtype=np.float32),
            np.asarray(data["heatmap"], dtype=np.float32),
            [_normalize_proposal(item) for item in data["proposals"]],
        )

    heatmap_file = run_dir / "point_score" / f"{tag}.npz"
    if not heatmap_file.exists():
        raise FileNotFoundError(f"AO-Grasp point_score file was not found: {heatmap_file}")
    data = np.load(heatmap_file, allow_pickle=True)["data"].item()
    return (
        np.asarray(data["pts"], dtype=np.float32),
        np.asarray(data["labels"], dtype=np.float32),
        [_load_single_proposal(run_dir)],
    )


def _load_single_proposal(run_dir: Path) -> Mapping[str, object]:
    proposal_path = run_dir / "proposal.json"
    if not proposal_path.exists():
        raise FileNotFoundError(
            "No proposal visualization data found. Expected either "
            f"{run_dir / 'grasp_proposals' / (run_dir.name + '.npz')} or {proposal_path}."
        )
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    return {
        "position_cam": proposal["position_fusion"],
        "quaternion_cam": proposal["quaternion_fusion"],
        "score": proposal["score"],
    }


def _load_body_frame(run_dir: Path) -> Mapping[str, object] | None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ("fusion_frame_position_reference", "fusion_frame_quat_reference")
    if not all(key in metadata for key in required):
        return None
    return {key: metadata[key] for key in required}


def _normalize_proposal(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        if "position_cam" in value and "quaternion_cam" in value:
            return value
        if "position_fusion" in value and "quaternion_fusion" in value:
            return {
                "position_cam": value["position_fusion"],
                "quaternion_cam": value["quaternion_fusion"],
                "score": value["score"],
            }
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        quat_xyzw = np.asarray(value[1], dtype=np.float32).reshape(4)
        return {
            "position_cam": value[0],
            "quaternion_cam": [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])],
            "score": float(value[2]),
        }
    raise ValueError(f"Unsupported AO-Grasp proposal payload: {value!r}")


if __name__ == "__main__":
    main()
