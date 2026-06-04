"""Contact-GraspNet proposal service for the AO-Grasp sidecar stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Tuple


import numpy as np
from http_service import run_json_service


def _is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / 'config.yaml').exists()


def _resolve_checkpoint_dir(requested: Path) -> Path:
    requested = requested.expanduser()
    if _is_checkpoint_dir(requested):
        return requested

    basename = requested.name
    search_roots: list[Path] = []
    for candidate in (requested, requested.parent, Path('/models')):
        if candidate not in search_roots:
            search_roots.append(candidate)

    matches: list[Path] = []
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        if _is_checkpoint_dir(root):
            matches.append(root)
            continue
        for candidate in root.rglob(basename):
            if _is_checkpoint_dir(candidate):
                matches.append(candidate)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in matches:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)

    if len(deduped) == 1:
        return deduped[0]
    if len(deduped) > 1:
        def _rank(path: Path) -> tuple[int, int, str]:
            return (len(path.parts), len(path.name), str(path))
        return sorted(deduped, key=_rank)[0]

    raise FileNotFoundError(f"Contact-GraspNet checkpoint dir was not found: {requested}")


class ContactGraspNetBackend:
    def __init__(
        self,
        *,
        ao_grasp_root: Path,
        checkpoint_dir: Path,
        expected_points: int,
        num_heatmap_candidates: int,
    ) -> None:
        self.ao_grasp_root = ao_grasp_root
        self.checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
        self.expected_points = int(expected_points)
        self.num_heatmap_candidates = int(num_heatmap_candidates)

        if not self.ao_grasp_root.exists():
            raise FileNotFoundError(f"AO-Grasp source root was not found: {self.ao_grasp_root}")
        if not (self.checkpoint_dir / 'config.yaml').exists():
            raise FileNotFoundError(
                f"Contact-GraspNet checkpoint dir is missing config.yaml: {self.checkpoint_dir}"
            )

        outer_root = self.ao_grasp_root / 'contact_graspnet'
        inner_root = outer_root / 'contact_graspnet'
        for entry in (str(outer_root), str(inner_root)):
            if entry in sys.path:
                sys.path.remove(entry)
        # Keep the inner module directory ahead of the package root so the config value
        # `model: contact_graspnet` resolves to `contact_graspnet.py`, not the package dir.
        sys.path.insert(0, str(inner_root))
        sys.path.insert(1, str(outer_root))

        import tensorflow.compat.v1 as tf
        import config_utils
        from contact_grasp_estimator import GraspEstimator

        self._tf = tf
        self._tf.disable_eager_execution()

        global_config = config_utils.load_config(str(self.checkpoint_dir), batch_size=1)
        global_config['TEST']['second_thres'] = 0.10
        global_config['DATA']['raw_num_points'] = self.expected_points

        self._grasp_estimator = GraspEstimator(global_config)
        self._grasp_estimator.build_network()

        saver = tf.train.Saver(save_relative_paths=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self._session = tf.Session(config=config)
        self._grasp_estimator.load_weights(self._session, saver, str(self.checkpoint_dir), mode='test')

    def infer(self, points_cam: np.ndarray, heatmap: np.ndarray) -> list:
        if points_cam.shape != (self.expected_points, 3):
            raise ValueError(
                f"Expected points_cam shape {(self.expected_points, 3)}, got {tuple(points_cam.shape)}."
            )
        if heatmap.shape != (self.expected_points,):
            raise ValueError(
                f"Expected heatmap shape {(self.expected_points,)}, got {tuple(heatmap.shape)}."
            )

        pred_grasps_cf, scores, contact_pts, gripper_openings = self._grasp_estimator.predict_scene_grasps(
            self._session,
            points_cam,
            pc_segments={},
            local_regions=False,
            filter_grasps=False,
            forward_passes=1,
            pred_full=True,
        )
        pred_grasps = pred_grasps_cf[-1]
        if pred_grasps is None or len(pred_grasps) == 0:
            return []

        target_positions = np.asarray(
            [self._get_gripper_target_pos_from_cgn_grasp(grasp) for grasp in pred_grasps],
            dtype=np.float32,
        )
        top_k_indices = np.argsort(heatmap)[-self.num_heatmap_candidates :]
        top_k_points = points_cam[top_k_indices]
        from scipy.spatial import KDTree

        closest_indices = KDTree(target_positions).query(top_k_points)[1]

        proposals = []
        seen_indices = set()
        for heatmap_rank, grasp_index in enumerate(closest_indices.tolist()):
            grasp_index = int(grasp_index)
            if grasp_index in seen_indices:
                continue
            seen_indices.add(grasp_index)
            proposals.append(
                {
                    "position_cam": target_positions[grasp_index].tolist(),
                    "quaternion_cam": self._get_gripper_ori_from_cgn_grasp(pred_grasps[grasp_index]).tolist(),
                    "score": float(heatmap[top_k_indices[heatmap_rank]]),
                }
            )

        proposals.sort(key=lambda item: -item['score'])
        return proposals

    def healthcheck(self) -> dict:
        return {
            "status": "ok",
            "expected_points": self.expected_points,
            "num_heatmap_candidates": self.num_heatmap_candidates,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

    @staticmethod
    def _get_gripper_target_pos_from_cgn_grasp(cgn_grasp: np.ndarray) -> np.ndarray:
        control_point = np.array([[0.0, 0.0, 1.0527314e-01]], dtype=np.float32)
        rotated = np.matmul(control_point, cgn_grasp[:3, :3].T)
        target_pos = rotated + np.expand_dims(cgn_grasp[:3, 3], 0)
        return np.squeeze(target_pos).astype(np.float32)

    @staticmethod
    def _get_gripper_ori_from_cgn_grasp(cgn_grasp: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        grasp_euler = Rotation.from_matrix(cgn_grasp[:3, :3]).as_euler('zxy', degrees=False)
        grasp_euler[0] += np.pi / 2.0
        quat_xyzw = Rotation.from_euler('zxy', grasp_euler).as_quat().astype(np.float32)
        return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


class ContactGraspNetService:
    def __init__(self, backend: ContactGraspNetBackend) -> None:
        self._backend = backend

    def healthz(self, _payload: Optional[dict]) -> Tuple[int, dict]:
        return 200, self._backend.healthcheck()

    def proposals(self, payload: Optional[dict]) -> Tuple[int, dict]:
        payload = payload or {}
        request_id = payload.get('request_id')
        points_cam = np.asarray(payload.get('points_cam'), dtype=np.float32)
        heatmap = np.asarray(payload.get('heatmap'), dtype=np.float32)
        proposals = self._backend.infer(points_cam, heatmap)
        return 200, {
            'request_id': request_id,
            'proposals': proposals,
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='AO-Grasp Contact-GraspNet sidecar service')
    parser.add_argument('--host', type=str, default=os.environ.get('CGN_HOST', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('CGN_PORT', '8002')))
    parser.add_argument(
        '--expected-points',
        type=int,
        default=int(os.environ.get('CGN_EXPECTED_POINTS', '16384')),
    )
    parser.add_argument(
        '--num-heatmap-candidates',
        type=int,
        default=int(os.environ.get('CGN_NUM_HEATMAP_CANDIDATES', '200')),
    )
    parser.add_argument(
        '--ao-grasp-root',
        type=Path,
        default=Path(os.environ.get('AO_GRASP_ROOT', '/opt/ao-grasp')),
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=Path(
            os.environ.get(
                'CGN_CHECKPOINT_DIR',
                '/models/scene_test_2048_bs3_hor_sigma_001',
            )
        ),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    backend = ContactGraspNetBackend(
        ao_grasp_root=args.ao_grasp_root,
        checkpoint_dir=args.checkpoint_dir,
        expected_points=args.expected_points,
        num_heatmap_candidates=args.num_heatmap_candidates,
    )
    service = ContactGraspNetService(backend)
    run_json_service(
        args.host,
        args.port,
        {
            ('GET', '/healthz'): service.healthz,
            ('POST', '/proposals'): service.proposals,
        },
    )


if __name__ == '__main__':
    main()
