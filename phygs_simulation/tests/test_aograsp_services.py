from __future__ import annotations

import numpy as np

from cgn_service import ContactGraspNetService
from pointscore_service import PointScoreService


class _FakePointScoreBackend:
    def healthcheck(self) -> dict:
        return {'status': 'ok', 'kind': 'pointscore'}

    def infer(self, points_cam: np.ndarray) -> np.ndarray:
        return np.sum(points_cam, axis=1, dtype=np.float32)


class _FakeCgnBackend:
    def healthcheck(self) -> dict:
        return {'status': 'ok', 'kind': 'cgn'}

    def infer(self, points_cam: np.ndarray, heatmap: np.ndarray) -> list[dict]:
        return [
            {
                'position_cam': points_cam[0].tolist(),
                'quaternion_cam': [1.0, 0.0, 0.0, 0.0],
                'score': float(heatmap[0]),
            }
        ]


def test_pointscore_service_contract() -> None:
    service = PointScoreService(_FakePointScoreBackend())
    status, health = service.healthz(None)
    assert status == 200
    assert health['status'] == 'ok'

    payload = {
        'request_id': 'abc123',
        'points_cam': [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    }
    status, response = service.heatmaps(payload)
    assert status == 200
    assert response['request_id'] == 'abc123'
    assert response['labels'] == [1.0, 1.0]


def test_cgn_service_contract() -> None:
    service = ContactGraspNetService(_FakeCgnBackend())
    status, health = service.healthz(None)
    assert status == 200
    assert health['status'] == 'ok'

    payload = {
        'request_id': 'req-1',
        'points_cam': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'heatmap': [0.9, 0.1],
    }
    status, response = service.proposals(payload)
    assert status == 200
    assert response['request_id'] == 'req-1'
    assert len(response['proposals']) == 1
    assert abs(response['proposals'][0]['score'] - 0.9) < 1e-6
