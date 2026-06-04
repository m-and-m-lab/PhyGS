# Docker

This directory contains Docker integration for IsaacLab/cuRobo compatibility and AO-Grasp sidecar services.

## Files

| Path | Purpose |
| --- | --- |
| `docker-compose.curobofix.patch.yaml` | Compose patch used when launching Isaac Lab with the cuRobo-compatible Dockerfile. |
| `Dockerfile.curobofix` | IsaacLab container customization for cuRobo-related dependencies. |
| `compose.aograsp.yml` | Runs AO-Grasp pointscore and CGN services. |
| `aograsp/pointscore.Dockerfile` | Builds the pointscore inference service. |
| `aograsp/cgn.Dockerfile` | Builds the Contact-GraspNet proposal service. |
| `aograsp/services/` | Lightweight HTTP service wrappers. |
| `artifacts/ao-grasp-models/` | Expected local mount point for AO-Grasp model artifacts. |

## IsaacLab Container Patch

From the IsaacLab Docker directory:

```bash
cd /workspace/isaaclab/docker
./container.py start base --files ../scripts/phygs_simulation/docker/docker-compose.curobofix.patch.yaml
./container.py enter base
```

Inside the container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -i
```

## AO-Grasp Services

Start sidecars:

```bash
cd /workspace/isaaclab/scripts/phygs_simulation
docker compose -f docker/compose.aograsp.yml up --build
```

Services:

| Service | Container port | Default host port | Health endpoint |
| --- | --- | --- | --- |
| `ao-pointscore` | `8001` | `18081` | `/healthz` |
| `ao-cgn` | `8002` | `18082` | `/healthz` |

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AO_GRASP_VISIBLE_DEVICES` | `all` | GPU visibility for both services. |
| `POINTSCORE_DEVICE` | `cuda:0` | Torch device used by pointscore. |
| `AO_GRASP_EXPECTED_POINTS` | `16384` | Point-count contract shared by PointScore, CGN, and `ao_grasp.total_points`. |
| `AO_POINTSCORE_PORT` | `18081` | Host port for pointscore. |
| `AO_CGN_PORT` | `18082` | Host port for CGN. |
| `AO_CGN_CHECKPOINT_DIR` | `/models/scene_test_2048_bs3_hor_sigma_001` | Model directory inside the CGN container. |
| `AO_GRASP_MODELS_DIR` | `./artifacts/ao-grasp-models` | Host model directory mounted into `/models`. |

## Network Notes

If IsaacLab runs in the same Docker network as AO-Grasp, use service names:

```yaml
pointscore_url: http://ao-pointscore:8001
cgn_url: http://ao-cgn:8002
```

If IsaacLab runs on the host or in a different container network, use exposed host ports:

```yaml
pointscore_url: http://127.0.0.1:18081
cgn_url: http://127.0.0.1:18082
```

## Rules

- Do not bake model artifacts into benchmark source commits.
- Keep sidecar service APIs small and explicit.
- Healthchecks must fail when models or GPU runtime are unavailable.
- Do not replace service failures with fake grasps in benchmark scripts.
