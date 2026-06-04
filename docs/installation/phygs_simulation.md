# Installing phygs_simulation (IsaacLab benchmark)

`phygs_simulation` is the Spot benchmark stack: Spot USD/URDF assets, a Spot-SDK-shaped skills API, learned locomotion policy, AO-Grasp grasp proposal sidecars, and the cuRobo motion planner. It runs **inside the IsaacLab Docker container** and depends on the vendored `ao-grasp` and `curobo` git submodules under `phygs_simulation/third_party/`.

## Validated versions

This install path has been exercised against:

| Component | Version |
| --- | --- |
| IsaacSim | **5.1.0** |
| IsaacLab | **2.3.0** |

Other versions may work but are untested. The `Dockerfile.curobofix` patch was authored against the IsaacSim 5.1.0 base image layout (including the `omni.services.pip_archive` packaging shim it restores).

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Linux x86_64 host | The IsaacLab Docker workflow does not support macOS/Windows natively. |
| NVIDIA GPU | Required for IsaacSim, IsaacLab, cuRobo, and AO-Grasp inference. |
| NVIDIA driver + NVIDIA container runtime | Docker GPU passthrough is mandatory. |
| Docker + Docker Compose v2 | Used for both the IsaacLab container build and the AO-Grasp sidecars. |
| PhyGS checkout with submodules initialized | See the [installation overview](./README.md#clone-the-repo). |

## Step 1 — clone PhyGS and initialize submodules

```bash
git clone --recursive https://github.com/m-and-m-lab/PhyGS.git
cd PhyGS

# Or, on an existing checkout:
git submodule update --init --recursive
```

Verify the submodules populated:

```bash
ls phygs_simulation/third_party/ao-grasp/aograsp/aograsp_model/  # should list conf.pth and 770-network.pth
ls phygs_simulation/third_party/ao-grasp/contact_graspnet/contact_graspnet/contact_grasp_estimator.py
ls phygs_simulation/third_party/curobo/pyproject.toml
```

## Step 2 — clone IsaacLab and mount phygs_simulation

Clone IsaacLab 2.3.0 separately and mount `phygs_simulation` into its scripts tree under that exact name — the docker compose patch (`docker-compose.curobofix.patch.yaml`) expects to find `scripts/phygs_simulation/docker/Dockerfile.curobofix`.

```bash
# Clone IsaacLab 2.3.0 alongside PhyGS.
git clone --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/scripts

# Symlink the phygs_simulation directory from this PhyGS checkout into IsaacLab/scripts/.
ln -s /absolute/path/to/PhyGS/phygs_simulation phygs_simulation
```

Use an absolute path in the symlink so the Docker build context resolves correctly. A copy works too, but a symlink avoids a second source-of-truth.

## Step 3 — build and start the patched IsaacLab container

The `Dockerfile.curobofix` patch builds on the IsaacLab base image and adds:

- CUDA 12.8 toolkit + cuDNN.
- cuRobo installed in editable mode from the `third_party/curobo` submodule (copied into the IsaacLab tree by the Docker build context).
- nvblox v0.0.9 and `nvblox_torch` built against the IsaacSim-bundled Torch.
- Open3D, Pillow, and matplotlib for AO-Grasp debug artifacts.
- A handful of packaging shims for known IsaacSim 5.1.0 / cuRobo dependency conflicts.

```bash
cd IsaacLab/docker

# Build and start the base IsaacLab container with the PhyGS compose patch.
./container.py start base --files ../scripts/phygs_simulation/docker/docker-compose.curobofix.patch.yaml

# Drop into the running container.
./container.py enter base
```

The first build is slow (CUDA toolkit + nvblox build); subsequent rebuilds use cached layers.

## Step 4 — install IsaacLab inside the container

Inside the container shell:

```bash
cd /workspace/isaaclab
./isaaclab.sh -i
```

If you see a pip dependency-resolver warning about `rl-games`/`psutil`, ignore it — it's harmless and upstream-known.

## Step 5 — install `phygs_simulation` as an editable package (optional but recommended)

Still inside the container:

```bash
cd /workspace/isaaclab/scripts/phygs_simulation
python -m pip install -e .
```

This registers the `helpers` and `skills` packages on `sys.path` for `python -c "from skills.spot import ..."` style imports. Tests do not require this step — `tests/conftest.py` adjusts `sys.path` directly.

## Step 6 — start the AO-Grasp sidecars

AO-Grasp ships as two sidecar HTTP services. Build context is the `phygs_simulation/` directory, so both run on the host (or any Docker-capable environment with access to this checkout):

```bash
cd /absolute/path/to/PhyGS/phygs_simulation
docker compose -f docker/compose.aograsp.yml up --build
```

The services expose:

| Service | Container endpoint | Default host port | Healthcheck |
| --- | --- | --- | --- |
| `ao-pointscore` | `http://ao-pointscore:8001` | `18081` | `/healthz` |
| `ao-cgn` | `http://ao-cgn:8002` | `18082` | `/healthz` |

The CGN sidecar mounts a model directory at container path `/models`. The default mount point is `./artifacts/ao-grasp-models/` (host-relative to the compose file). Place your CGN checkpoint directory there or override `AO_GRASP_MODELS_DIR` / `AO_CGN_CHECKPOINT_DIR`.

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AO_GRASP_VISIBLE_DEVICES` | `all` | GPU visibility for both services. |
| `POINTSCORE_DEVICE` | `cuda:0` | Torch device for pointscore. |
| `AO_GRASP_EXPECTED_POINTS` | `16384` | Point-count contract shared by pointscore, CGN, and the `ao_grasp.total_points` client setting. |
| `AO_POINTSCORE_PORT` | `18081` | Host port for pointscore. |
| `AO_CGN_PORT` | `18082` | Host port for CGN. |
| `AO_CGN_CHECKPOINT_DIR` | `/models/scene_test_2048_bs3_hor_sigma_001` | Model directory inside the CGN container. |
| `AO_GRASP_MODELS_DIR` | `./artifacts/ao-grasp-models` | Host model directory mounted into `/models`. |

### Network reachability

- **Same Docker network:** point client configs at `http://ao-pointscore:8001` / `http://ao-cgn:8002`.
- **IsaacLab on the host (or a different Docker network):** use the published host ports — `http://127.0.0.1:18081` / `http://127.0.0.1:18082`.

## Step 7 — verify the install

### Pure Python unit tests (no IsaacSim app required)

```bash
cd /workspace/isaaclab/scripts/phygs_simulation
pytest -q tests
```

Narrower SDK-only subset (skips AO-Grasp/CuRobo/Torch when those backends aren't reachable):

```bash
pytest -q tests/test_spot_sdk_api.py tests/test_spot_manipulation_api.py
```

### Locomotion smoke test (WASD)

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/phygs_simulation/tests/isaaclab_test/spot_locomotion_wasd.py
```

### Drawer manipulation smoke test (requires AO-Grasp sidecars up)

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/phygs_simulation/tests/isaaclab_test/spot_manipulation_drawer.py --enable_cameras
```

Successful completion drives the manipulation feedback lifecycle to `DONE`. Missing AO-Grasp services, missing camera depth frames, missing target AABB, or cuRobo planning failure should each surface as a clear non-zero error — `phygs_simulation` deliberately fails fast rather than silently substituting fallbacks.

### General interactive scene

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/phygs_simulation/scripts/interactive_search.py \
  --enable_cameras \
  --robot_usd scripts/phygs_simulation/spot_model/spot_arm_w_cam.usd \
  --scene_usd /absolute/path/to/your/export_scene.usdc
```

Use a scene USD exported from `scene_generation/infinigen/` (see [`scene_generation.md`](./scene_generation.md)) or any compatible USD asset.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `Dockerfile.curobofix` build fails copying `third_party/curobo`. | Submodules not initialized in this PhyGS checkout — re-run `git submodule update --init --recursive` before `container.py start`. |
| AO-Grasp build fails on `compile_pointnet_tfops.sh`. | Nested `ao-grasp/contact_graspnet/` submodule not initialized. Re-run the recursive submodule update. |
| `pip install -e .` succeeds but `from skills.spot import …` still fails outside `tests/`. | Run from `phygs_simulation/`, not from `/workspace/isaaclab`. The setup.py uses `package_dir={"": "scripts"}`, so packages live under `scripts/`. |
| Manipulation smoke test reports `RuntimeError: Failed to resolve target AABB` | Drawer target prim is not visible to the configured Spot cameras. Adjust the smoke-test YAML's drawer pose; do not narrow the target prim path. |
| Camera depth frames are missing. | `--enable_cameras` not passed, or the IsaacSim build does not have camera support. The smoke tests intentionally fail fast in this case. |

## Where to go next

- `phygs_simulation/README.md` — quick-start commands and benchmark pipeline.
- `phygs_simulation/ARCHITECTURE.md` — Spot SDK facade, IsaacLab binder, manipulation lifecycle, cuRobo and AO-Grasp pipelines.
- `phygs_simulation/AGENTS.md` — coding rules (fail-fast, no silent fallbacks, SDK-shaped public API).
- `phygs_simulation/PLANS.md` — milestones and known gaps.
