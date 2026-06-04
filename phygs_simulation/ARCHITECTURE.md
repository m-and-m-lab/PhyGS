# Architecture

This document describes the Interactive Search benchmark architecture, with emphasis on the Spot SDK facade, IsaacLab backends, AO-Grasp pipeline, CuRobo planning pipeline, and smoke-test lifecycle.

## System Overview

Interactive Search is organized around four layers:

| Layer | Responsibility | Representative modules |
| --- | --- | --- |
| Benchmark launchers | Build scenes, load configs, run episodes, collect terminal state. | `scripts/interactive_search.py`, `tests/isaaclab_test/*.py` |
| SDK-shaped API | Expose Spot-like robot clients and command builders. | `scripts/skills/spot/sdk.py`, `scripts/skills/spot/robot_command.py` |
| Skill implementations | Locomotion, manipulation, perception preparation, and planner control. | `scripts/skills/locomotion`, `scripts/skills/manipulation` |
| Simulation and services | IsaacLab articulation/camera backends, AO-Grasp sidecars, CuRobo, USD assets. | `scripts/skills/spot/isaaclab_backend.py`, `docker/aograsp`, `spot_model` |

The key architectural boundary is the SDK facade. Benchmark scripts should talk to `Robot` clients. IsaacLab details should stay behind service backends and binders.

## High-Level Data Flow

```text
YAML config
  |
  v
IsaacLab launch script
  |
  v
InteractiveScene: Spot + cameras + task assets
  |
  v
create_standard_sdk -> Robot -> ensure_client(...)
  |
  v
Lease, power, state, image, robot-command, manipulation clients
  |
  v
Manipulation request: target prim + camera names + target center
  |
  v
RGB-D observations from mounted cameras
  |
  v
AO-Grasp point-cloud preparation and sidecar inference
  |
  v
Grasp proposal -> pregrasp/grasp EE targets
  |
  v
CuRobo local collision world and trajectory planning
  |
  v
IsaacLab arm and gripper joint targets
  |
  v
Manipulation feedback: planning/executing/grasping/done/failed
```

## Spot SDK Facade

The SDK facade lives in `scripts/skills/spot`.

| Type | Role |
| --- | --- |
| `Sdk` | Registry that creates and caches simulation-backed `Robot` instances by address. |
| `Robot` | Holds service factories, client cache, power/auth backend, and `time_sync`. |
| `LeaseClient` | Acquires, retains, lists, and returns simulation leases. |
| `LeaseKeepAlive` | Background lease retention helper, mirroring the Spot SDK pattern. |
| `RobotCommandClient` | Submits mobility, arm, and gripper commands through a backend. |
| `RobotStateClient` | Returns current robot state snapshots. |
| `ImageClient` | Lists camera sources and returns image payloads. |
| `ManipulationApiClient` | Submits manipulation requests and polls command feedback. |

The facade intentionally does not implement full protobuf or gRPC parity. It matches Spot SDK naming and programming style closely enough for benchmark code to look like real robot-client code.

## Service Resolution

`Robot.ensure_client(service_name)` resolves service names through installed factories.

Supported service names:

| Name | Client |
| --- | --- |
| `lease` | `LeaseClient` |
| `robot-command` | `RobotCommandClient` |
| `robot-state` | `RobotStateClient` |
| `image` | `ImageClient` |
| `manipulation` | `ManipulationApiClient` |

Aliases such as `robot_state` and `manipulation-api` normalize to canonical service names.

## IsaacLab Binding

`scripts/skills/spot/isaaclab_backend.py` connects a `Robot` facade to live IsaacLab objects.

`bind_isaaclab_spot_robot(...)` installs:

| Backend | Function |
| --- | --- |
| `InMemoryLeaseBackend` | Tracks the active body lease for the simulated robot. |
| `InMemoryRobotControlBackend` | Tracks authentication, power state, estop state, and lease requirement. |
| `IsaacLabRobotCommandBackend` | Applies arm and gripper joint targets directly to the IsaacLab articulation. |
| `IsaacLabRobotStateBackend` | Proxies the manipulation runner robot-state snapshot. |
| `IsaacLabImageBackend` | Captures mounted camera frames and poses. |
| `IsaacLabManipulationApiBackend` | Wraps `IsaacLabManipulationRunner` in command/feedback lifecycle semantics. |

The binder returns `IsaacLabSpotServiceBinder`. Launch scripts call `binder.step()` each simulation tick so active manipulation commands advance.

## Manipulation Lifecycle

Manipulation uses a single-active-command lifecycle:

1. `ManipulationApiClient.manipulation_api_command(request)` submits a request.
2. The backend assigns a `command_id`.
3. If another manipulation command is active, it is marked `OVERRIDDEN`.
4. The runner attempts to trigger AO-Grasp and CuRobo planning.
5. Feedback starts as `PLANNING` after successful trigger.
6. Each `binder.step()` advances the runner.
7. Feedback maps skill status into SDK-level states.
8. Terminal states are `DONE`, `FAILED`, and `OVERRIDDEN`.

Feedback state mapping:

| Skill status | Manipulation feedback |
| --- | --- |
| `planning` | `PLANNING` |
| `executing` | `EXECUTING` |
| `grasping` | `GRASPING` |
| `done` | `DONE` |
| `failed` | `FAILED` |
| replaced active command | `OVERRIDDEN` |

## AO-Grasp Pipeline

AO-Grasp integration lives in one module plus the shared helper packages.

| Module | Responsibility |
| --- | --- |
| `aograsp_client.py` | AO-facing point-cloud preparation, debug artifacts, HTTP healthcheck, pointscore request, CGN proposal request, response decoding. |
| `grasp_types.py` | Shared request, observation, snapshot, proposal, and execution-target dataclasses. |
| `docker/aograsp/services` | HTTP service wrappers around pointscore and CGN inference. |

For drawer manipulation, the benchmark computes the world-space AABB of the drawer root prim, converts its center into the robot base frame, and passes that as `target_center_b`. This is intentional. The benchmark should not silently fall back to sublinks such as `link_1`.

## CuRobo Planning Pipeline

CuRobo integration lives mostly in `curobo_joint_command.py` and `manipulation_skill.py`.

Major responsibilities:

| Component | Responsibility |
| --- | --- |
| `CuroboJointCommandPlanner` | Owns motion-generation config, planner state, joint command extraction, and collision-world updates. |
| `DepthCollisionWorldConfig` | Configures depth and local ESDF collision world behavior. |
| `CuroboController` | Converts robot state and target world pose into CuRobo planning calls. |
| `ManipulationSkill` | Owns skill lifecycle, queued arm trajectory execution, and gripper close/open behavior. |
| `IsaacLabManipulationRunner` | Bridges skill commands to IsaacLab joint targets and handles AO-Grasp-triggered sequences. |

Planning inputs:

- Current joint positions and velocities.
- Robot root pose.
- End-effector body pose.
- Target pregrasp and grasp pose.
- Stage collision world or depth/local ESDF collision world.
- CuRobo robot config from `spot_model/configuration/spot_arm_curobo.yaml`.

Planning outputs:

- Ordered joint targets for Spot arm joints.
- Gripper position targets.
- Feedback status for manipulation lifecycle.

## Locomotion Pipeline

Locomotion uses a learned policy checkpoint and a Spot command backend.

| Module | Role |
| --- | --- |
| `skills/locomotion/spot_policy.py` | Loads policy config, maps runtime joints to policy joints, and runs policy inference. |
| `skills/locomotion/isaaclab_spot_backend.py` | Builds articulation config, releases the root joint for locomotion, and converts velocity/stand/sit commands into policy control. |
| `skills/locomotion/api.py` | Provides `SpotLocomotionClient` convenience methods. |
| `tests/isaaclab_test/spot_locomotion_wasd.py` | Manual smoke launcher with terminal control. |

Locomotion currently shares command-builder types with manipulation but uses a dedicated backend that accepts mobility commands only.

## Scene and Asset Layer

Assets come from several sources:

| Source | Usage |
| --- | --- |
| `spot_model` | Spot robot, cameras, URDF/USD, CuRobo config. |
| `.scene_usd/articulations` under the broader IsaacLab scripts tree | Task assets such as drawers. |
| `docs/assets` | Generated or curated scene assets used during development. |
| `third_party/infinigen` | Scene/object generation source. |
| `scripts/generate` and `scripts/blender` | Asset generation, Blender conversion, and scene export helpers. |

Benchmark launchers should reference assets explicitly through YAML config when possible.

## Smoke Test Architecture

Smoke tests under `tests/isaaclab_test` are full application launchers.

| Smoke script | Purpose |
| --- | --- |
| `spot_locomotion_wasd.py` | Spawn Spot, release root joint, load locomotion policy, and manually command motion. |
| `spot_manipulation_drawer.py` | Spawn fixed-base Spot, ground plane, drawer, cameras, AO-Grasp/CuRobo runner, and execute one SDK-backed manipulation command. |

Smoke scripts are intentionally stricter than demos:

- They validate config paths.
- They verify services before execution.
- They verify cameras before planning.
- They verify target AABB resolution.
- They exit with a non-zero error on missing prerequisites or failed planning.

## Test Layers

| Test type | Location | Runtime |
| --- | --- | --- |
| Pure Python unit tests | `tests/test_*.py` | Standard Python environment with local dependencies. |
| Service tests | AO-Grasp tests | Require sidecar services for smoke coverage. |
| IsaacLab smoke tests | `tests/isaaclab_test/*.py` | Require Isaac Sim, Isaac Lab, GPU, assets, and camera support. |

Pure tests should be kept narrow and deterministic. IsaacLab smoke tests should validate integration and runtime assumptions.

## Failure Model

The benchmark should fail clearly for:

- Missing robot, drawer, policy, or CuRobo config files.
- Missing AO-Grasp services.
- Missing camera sources or missing depth frames.
- Missing target AABB.
- Missing arm, gripper, or end-effector joints/bodies.
- Missing CUDA or planner runtime dependencies.
- AO-Grasp returning no proposals.
- CuRobo failing to plan a queued target.
- Manipulation timeout or terminal failure feedback.

The benchmark should not silently patch over these failures.

## Extension Points

To add a new skill:

1. Add core dataclasses or request types under `scripts/skills`.
2. Add a Spot-SDK-shaped client or command builder if it is robot-facing.
3. Add the IsaacLab backend or binder code separately.
4. Add pure unit tests for API behavior.
5. Add an IsaacLab smoke script if the skill touches simulation.

To add a new benchmark task:

1. Add a task YAML with robot, scene, target, and timeout sections.
2. Reference concrete USD assets and target prim paths.
3. Add task-specific success criteria.
4. Save result artifacts.
5. Add documentation and a smoke command.
