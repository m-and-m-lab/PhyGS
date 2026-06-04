# Tests

This directory contains two kinds of tests: pure Python unit tests and full IsaacLab smoke launchers.

## Test Layout

| Path | Purpose |
| --- | --- |
| `test_spot_command_api.py` | Command builders, feedback objects, and blocking command helpers. |
| `test_spot_sdk_api.py` | SDK registry, client resolution, lease keepalive, and power lifecycle. |
| `test_spot_locomotion_api.py` | Locomotion client wrapper behavior. |
| `test_spot_manipulation_api.py` | Manipulation feedback lifecycle, image backend, and state backend. |
| `test_spot_policy_remap.py` | Runtime-to-policy joint-name remapping. |
| `test_aograsp_client.py` | AO-Grasp HTTP client request/response validation. |
| `test_aograsp_pipeline.py` | AO-Grasp point-cloud preparation and target-selection helpers. |
| `test_aograsp_services.py` | AO-Grasp service wrapper behavior. |
| `test_aograsp_smoke.py` | AO-Grasp integration smoke coverage. |
| `test_curobo_debug_viz.py` | Offline CuRobo ESDF/depth/plan debug artifact generation. |
| `test_curobo_joint_command.py` | CuRobo joint-command helper behavior. |
| `isaaclab_test/` | Full IsaacLab launch scripts and YAML configs. |

## Pure Unit Tests

Run from the project root:

```bash
cd /workspace/isaaclab/scripts/phygs_simulation
pytest -q tests
```

If the local shell does not have Torch, IsaacLab, or CuRobo installed, run the narrower pure-SDK tests:

```bash
pytest -q tests/test_spot_sdk_api.py tests/test_spot_manipulation_api.py
```

## IsaacLab Smoke Tests

Run smoke tests from the Isaac Lab root:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/phygs_simulation/tests/isaaclab_test/spot_locomotion_wasd.py
```

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/phygs_simulation/tests/isaaclab_test/spot_manipulation_drawer.py --enable_cameras
```

Smoke tests require:

| Requirement | Why |
| --- | --- |
| Isaac Sim and Isaac Lab | They launch a real simulator application. |
| NVIDIA GPU | Required for rendering, simulation, and planning stack. |
| Camera support | Manipulation tests need mounted RGB-D frames. |
| AO-Grasp services | Drawer manipulation submits real proposal requests. |
| CuRobo runtime | Arm planning is required for manipulation success. |

## Test Philosophy

- Unit tests should isolate API contracts and deterministic helper logic.
- Smoke tests should validate real integration and fail loudly.
- Do not weaken tests with silent fallbacks.
- If a test cannot run in a lightweight environment, document the missing runtime instead of faking behavior.
