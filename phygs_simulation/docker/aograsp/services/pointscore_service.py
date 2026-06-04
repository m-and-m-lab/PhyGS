"""Pointscore service for the AO-Grasp sidecar stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np

from http_service import run_json_service


class PointScoreBackend:
    def __init__(
        self,
        *,
        ao_grasp_root: Path,
        model_conf_path: Path,
        checkpoint_path: Path,
        device: str,
        expected_points: int,
    ) -> None:
        self.ao_grasp_root = ao_grasp_root
        self.model_conf_path = model_conf_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.expected_points = int(expected_points)

        if not self.ao_grasp_root.exists():
            raise FileNotFoundError(f"AO-Grasp source root was not found: {self.ao_grasp_root}")
        if not self.model_conf_path.exists():
            raise FileNotFoundError(f"Pointscore config was not found: {self.model_conf_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Pointscore checkpoint was not found: {self.checkpoint_path}")

        sys.path.insert(0, str(self.ao_grasp_root))
        sys.path.insert(0, str(self.ao_grasp_root / 'aograsp' / 'models' / 'Pointnet2_PyTorch'))

        import torch
        import aograsp.model_utils as model_utils

        self._torch = torch
        requested_device = str(self.device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"POINTSCORE_DEVICE={requested_device} but torch.cuda.is_available() is False inside the ao-pointscore "
                "container. Ensure the NVIDIA runtime is exposing GPUs to Docker, or set POINTSCORE_DEVICE=cpu for "
                "debugging only."
            )
        map_location = requested_device if requested_device == "cpu" or requested_device.startswith("cuda") else None
        self._model = model_utils.load_model(
            model_conf_path=str(self.model_conf_path),
            ckpt_path=str(self.checkpoint_path),
            map_location=map_location,
        )
        self._model.to(requested_device)
        self.device = requested_device
        self._model.eval()

    def infer(self, points_cam: np.ndarray) -> np.ndarray:
        if points_cam.shape != (self.expected_points, 3):
            raise ValueError(
                f"Expected points_cam shape {(self.expected_points, 3)}, got {tuple(points_cam.shape)}."
            )
        points_tensor = self._torch.from_numpy(points_cam.astype(np.float32, copy=False)).to(self.device)
        points_tensor = self._torch.unsqueeze(points_tensor, dim=0)
        with self._torch.no_grad():
            output = self._model.test({"pcs": points_tensor}, None)
        return output["point_score_heatmap"][0].detach().cpu().numpy().astype(np.float32)

    def healthcheck(self) -> dict:
        return {
            "status": "ok",
            "device": self.device,
            "expected_points": self.expected_points,
            "model_conf_path": str(self.model_conf_path),
            "checkpoint_path": str(self.checkpoint_path),
        }


class PointScoreService:
    def __init__(self, backend: PointScoreBackend) -> None:
        self._backend = backend

    def healthz(self, _payload: Optional[dict]) -> Tuple[int, dict]:
        return 200, self._backend.healthcheck()

    def heatmaps(self, payload: Optional[dict]) -> Tuple[int, dict]:
        payload = payload or {}
        request_id = payload.get("request_id")
        points_cam = np.asarray(payload.get("points_cam"), dtype=np.float32)
        labels = self._backend.infer(points_cam)
        return 200, {
            "request_id": request_id,
            "labels": labels.tolist(),
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AO-Grasp pointscore sidecar service")
    parser.add_argument("--host", type=str, default=os.environ.get("POINTSCORE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POINTSCORE_PORT", "8001")))
    parser.add_argument("--device", type=str, default=os.environ.get("POINTSCORE_DEVICE", "cuda:0"))
    parser.add_argument(
        "--expected-points",
        type=int,
        default=int(os.environ.get("POINTSCORE_EXPECTED_POINTS", "16384")),
    )
    parser.add_argument(
        "--ao-grasp-root",
        type=Path,
        default=Path(os.environ.get("AO_GRASP_ROOT", "/opt/ao-grasp")),
    )
    parser.add_argument(
        "--model-conf-path",
        type=Path,
        default=Path(os.environ.get("POINTSCORE_MODEL_CONF", "/opt/ao-grasp/aograsp/aograsp_model/conf.pth")),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(os.environ.get("POINTSCORE_MODEL_CKPT", "/opt/ao-grasp/aograsp/aograsp_model/770-network.pth")),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    backend = PointScoreBackend(
        ao_grasp_root=args.ao_grasp_root,
        model_conf_path=args.model_conf_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        expected_points=args.expected_points,
    )
    service = PointScoreService(backend)
    run_json_service(
        args.host,
        args.port,
        {
            ("GET", "/healthz"): service.healthz,
            ("POST", "/heatmaps"): service.heatmaps,
        },
    )


if __name__ == "__main__":
    main()
