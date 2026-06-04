from __future__ import annotations

import os

import pytest

from skills.manipulation import AoGraspClient, AoGraspServiceConfig


@pytest.mark.skipif(os.environ.get('AO_GRASP_SMOKE') != '1', reason='requires running AO-Grasp sidecars')
def test_aograsp_healthcheck_smoke() -> None:
    client = AoGraspClient(
        AoGraspServiceConfig(
            pointscore_url=os.environ.get('AO_POINTSCORE_URL', 'http://127.0.0.1:18081'),
            cgn_url=os.environ.get('AO_CGN_URL', 'http://127.0.0.1:18082'),
            timeout_s=float(os.environ.get('AO_TIMEOUT_S', '30.0')),
        )
    )
    health = client.healthcheck()
    assert health['pointscore']['status'] == 'ok'
    assert health['cgn']['status'] == 'ok'
