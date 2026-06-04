# Interactive Search

Interactive Search is a robotics benchmark workspace for evaluating embodied search, locomotion, perception, grasp proposal, and manipulation policies in Isaac Lab. The current benchmark centers on a simulated Boston Dynamics Spot robot with arm and mounted RGB-D cameras, a Spot-SDK-shaped skills API, AO-Grasp proposal services, and CuRobo motion generation.

The repository is intentionally structured as a benchmark stack, not a one-off demo. Scripts should fail fast when required assets, services, cameras, CUDA resources, or target geometry are missing. The public control surface should look like robot SDK code, while the implementation remains simulation-backed and testable.

## What This Project Does

This project provides:

| Area | Purpose |
| --- | --- |
| Spot simulation assets | Spot arm USDs, camera-mounted USDs, URDF/XRDF/CuRobo configuration, and learned locomotion policy assets. |
| Skills API | A small Spot-SDK-like Python facade for robot command, lease, power, state, image, locomotion, and manipulation clients. |
| Locomotion | IsaacLab-backed Spot-arm locomotion policy execution with a manual WASD smoke test. |
| Manipulation | AO-Grasp object proposal generation plus CuRobo arm planning and guarded grasp execution. |
| Perception | Mounted camera wrappers, RGB-D capture, target point-cloud preparation, and local ESDF collision world construction. |
| Benchmark smoke tests | IsaacLab scripts that launch full simulation scenes for locomotion and drawer manipulation validation. |
| Scene and asset generation | Infinigen-derived indoor scene assets, articulated objects, Blender utilities, and Isaac Sim conversion helpers. |
| AO-Grasp sidecars | Dockerized pointscore and Contact-GraspNet-style services used by the manipulation pipeline. |

## Repository Layout

| Path | Contents |
| --- | --- |
| `scripts/` | Standalone IsaacLab launch scripts, scene utilities, generation helpers, and the `skills` package. |
| `scripts/skills/spot/` | Spot-SDK-shaped facade: `create_standard_sdk`, `Robot`, command client, lease client, image/state clients, and IsaacLab binders. |
| `scripts/skills/locomotion/` | Spot locomotion policy wrapper and IsaacLab command backend. |
| `scripts/skills/manipulation/` | AO-Grasp client, RGB-D point-cloud pipeline, CuRobo planner wrapper, and manipulation runner. |
| `tests/` | Unit tests plus IsaacLab smoke launch scripts under `tests/isaaclab_test/`. |
| `docker/` | IsaacLab/cuRobo Docker patch and AO-Grasp sidecar service containers. |
| `spot_model/` | Spot USD, URDF, camera assets, and CuRobo robot configuration. |
| `policies/` | Learned Spot policy checkpoints used by simulation tests. |
| `docs/` | Generated asset catalog notes and scene-generation documentation. |
| `third_party/` | Vendor projects such as AO-Grasp and Infinigen. Treat these as upstream code unless explicitly patching integration points. |

## Prerequisites

Use a Linux workstation or server with:

| Requirement | Notes |
| --- | --- |
| NVIDIA GPU | Required for Isaac Sim, Isaac Lab, CuRobo, and AO-Grasp service inference. |
| NVIDIA driver and container runtime | Docker workflows expect GPU passthrough. |
| Isaac Lab workspace | This project is located under `isaaclab/scripts/interactive-search` and is launched with `./isaaclab.sh`. |
| Git submodules | Required for `third_party/ao-grasp`, `third_party/infinigen`, and Spot model dependencies. |
| Docker Compose | Required for AO-Grasp sidecars. |

## Installation

We use [IsaacLab Docker](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html) to provide a standardized environment for evaluation.

For interactive-search, use the included compose patch instead of editing docker/docker-compose.yaml by hand:

From the Isaac Lab root (remember to clone [isaaclab](https://github.com/isaac-sim/IsaacLab)):

```bash
cd IsaacLab/scripts
git clone --recursive https://github.com/m-and-m-lab/interactive-search.git -b zhenhao-spot-depth
cd IsaacLab/docker

# Build and start Isaac Lab with the patched cuRobo Dockerfile.
./container.py start base --files ../scripts/interactive-search/docker/docker-compose.curobofix.patch.yaml

# Enter the running container.
./container.py enter base
```

Inside the Isaac Lab container, install Isaac Lab if needed:

```bash
cd /workspace/isaaclab
./isaaclab.sh -i
```
**You might see the error below when installing, well, IGNORE it and wait until IsaacLab official fix it.
```bash
#ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. rl-games 1.6.1 requires psutil<6.0.0,>=5.9.0, but you have psutil 7.2.2 which is incompatible.
```

Install the project package in editable mode if you want direct imports:

```bash
cd /workspace/isaaclab/scripts/interactive-search
python -m pip install -e .
```

## Quick Start

Run the general interactive manipulation script:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/interactive-search/scripts/interactive_search.py \
  --enable_cameras \
  --robot_usd scripts/interactive-search/spot_model/spot_arm_w_cam.usd \
  --scene_usd <path/to/export_scene.usdc>
```

### Run the Spot locomotion WASD smoke test:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/interactive-search/tests/isaaclab_test/spot_locomotion_wasd.py
```

### Run the Spot drawer manipulation smoke test:

Start AO-Grasp services from the host or from a Docker-capable environment:

```bash
cd /workspace/isaaclab/scripts/interactive-search
docker compose -f docker/compose.aograsp.yml up --build
```

The AO-Grasp services expose:

| Service | Default host port | Container endpoint |
| --- | --- | --- |
| `ao-pointscore` | `18081` | `http://ao-pointscore:8001` |
| `ao-cgn` | `18082` | `http://ao-cgn:8002` |

If Isaac Lab runs outside the AO-Grasp Docker network, point configs at `http://127.0.0.1:18081` and `http://127.0.0.1:18082`.

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/interactive-search/tests/isaaclab_test/spot_manipulation_drawer.py --enable_cameras
```

<!-- ### Run unit tests that do not require a full Isaac Sim app:

```bash
cd /workspace/isaaclab/scripts/interactive-search
pytest -q tests/test_spot_command_api.py tests/test_spot_sdk_api.py tests/test_spot_manipulation_api.py
``` -->

## Benchmark Pipeline

The full manipulation benchmark follows this sequence:

1. IsaacLab launches a scene with Spot, camera sensors, a ground plane, and task assets.
2. The project creates a Spot-SDK-shaped robot via `create_standard_sdk(...)`.
3. IsaacLab service backends bind robot command, lease, power, state, image, and manipulation clients to the simulated articulation.
4. The benchmark acquires a lease, authenticates, powers on, and opens the gripper through the SDK-shaped API.
5. Mounted cameras provide RGB-D observations and camera poses.
6. AO-Grasp prepares a fused target-centered point cloud and calls pointscore plus grasp proposal sidecars.
7. The top grasp proposals are converted into pregrasp and grasp end-effector targets.
8. CuRobo builds a stage or local depth collision world and plans arm trajectories.
9. The manipulation runner streams arm and gripper commands to the IsaacLab articulation.
10. The manipulation API reports lifecycle feedback: planning, executing, grasping, done, failed, or overridden.

For deeper implementation details, read [ARCHITECTURE.md](ARCHITECTURE.md).

<!-- ## Coding Philosophy

This repository should stay robotics-benchmark clean:

| Rule | Meaning |
| --- | --- |
| Fail fast | Missing cameras, services, target AABBs, CUDA backends, invalid joints, and bad configs should raise clear errors. |
| No silent fallbacks | Do not hide missing data by switching targets, fake cameras, fake grasps, or relaxed planning modes. |
| SDK-shaped API | External scripts should use `sdk.create_robot(...)`, `robot.ensure_client(...)`, lease/power helpers, and service clients. |
| Clean and lean code | Keep wrappers thin, configs explicit, and backend responsibilities separated. |
| Benchmark repeatability | Prefer deterministic configs, named assets, stable target paths, and explicit smoke tests. |

Agent-specific rules are in [AGENTS.md](AGENTS.md). -->

## Key Commands

| Goal | Command |
| --- | --- |
| Run all lightweight unit tests | `pytest -q tests` |
| Run Spot locomotion smoke | `./isaaclab.sh -p scripts/interactive-search/tests/isaaclab_test/spot_locomotion_wasd.py` |
| Run Spot drawer manipulation smoke | `./isaaclab.sh -p scripts/interactive-search/tests/isaaclab_test/spot_manipulation_drawer.py --enable_cameras` |
| Start AO-Grasp sidecars | `docker compose -f docker/compose.aograsp.yml up --build` |
| Launch generic manipulation scene | `./isaaclab.sh -p scripts/interactive-search/scripts/interactive_search.py --enable_cameras ...` |

## Current Status

Implemented:

- Spot-SDK-shaped facade for command, lease, state, image, power, and manipulation services.
- Locomotion policy wrapper and WASD smoke test.
- AO-Grasp client, point-cloud preparation, target-centered fusion, and sidecar service Dockerfiles.
- CuRobo planner integration with stage and depth/local-ESDF collision updates.
- Drawer manipulation smoke test that targets the whole drawer asset root.
- Unit tests for SDK registry, command API, lease/power, manipulation feedback, AO-Grasp client/pipeline, and CuRobo helpers.

In progress:

- More benchmark tasks beyond drawer grasping.
- More robust scene/task generation metadata.
- Standardized metrics, logs, and replay artifacts.
- CI separation between pure Python tests and GPU IsaacLab smoke tests.

See [PLANS.md](PLANS.md) for milestones and backlog.
