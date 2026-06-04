# Agent Rules

This file defines how Codex agents and human contributors should work in this repository. The project is a robotics benchmark, so correctness, reproducibility, and clear failure modes matter more than convenience hacks.

## Core Principles

| Principle | Rule |
| --- | --- |
| Fail fast | If an asset, service, camera stream, target AABB, CUDA dependency, robot joint, or planner result is missing, raise a clear error immediately. |
| No silent fallback | Do not silently switch to another target, fake camera frame, manual point, placeholder grasp, CPU substitute, or relaxed planner mode. |
| SDK-shaped public API | Benchmark scripts should look like Spot SDK code: `create_standard_sdk`, `sdk.create_robot`, `robot.ensure_client`, lease, power, command, image, state, and manipulation clients. |
| Clean and lean code | Keep modules small, separate API facade from IsaacLab backend, avoid global state, and avoid broad compatibility shims. |
| Robotics benchmark mindset | Prefer explicit configs, deterministic target paths, named assets, measurable outputs, and repeatable smoke tests. |

## Required Workflow

1. Inspect local code before editing.
2. Preserve existing user changes and never revert unrelated files.
3. Make targeted edits only in the area needed for the task.
4. Add or update tests for API and behavior changes.
5. Run the narrowest useful tests first.
6. Report what was verified and what could not be run in the current environment.

## Coding Style

Use these standards:

| Area | Standard |
| --- | --- |
| Python | Typed dataclasses and explicit protocols are preferred for API boundaries. |
| Errors | Use `ValueError`, `RuntimeError`, `KeyError`, or specific service errors with actionable messages. |
| Config | Use YAML for launch configs and dataclasses for validated runtime config. |
| Imports | Keep heavy IsaacLab, CuRobo, and Torch imports behind optional package boundaries where possible. |
| Comments | Add comments only for robotics-specific reasoning or non-obvious control flow. |
| Fallbacks | Avoid defensive fallback trees. If a benchmark dependency is required, require it. |
| Naming | Prefer Spot SDK vocabulary for public clients and service methods. |

## SDK API Rules

The public robot-facing surface should remain close to the Spot SDK at the client and method-name level.

Required public patterns:

```python
sdk = create_standard_sdk("benchmark")
robot = sdk.create_robot("sim://spot", name="spot-sim")
lease_client = robot.ensure_client("lease")
command_client = robot.ensure_client("robot-command")
state_client = robot.ensure_client("robot-state")
image_client = robot.ensure_client("image")
manipulation_client = robot.ensure_client("manipulation")
```

Rules:

- Add new robot capabilities as service clients or builders, not as ad-hoc script methods.
- Keep the SDK facade simulation-backed and lightweight.
- Keep IsaacLab implementation details in backend/binder modules.
- Preserve existing locomotion call sites when extending command signatures.
- Use command ids and feedback objects for long-running actions.
- Only one active manipulation command should run at a time unless a future milestone explicitly adds queuing.

## Fail-Fast Examples

Acceptable:

```python
if bounds is None:
    raise RuntimeError(f"Failed to resolve target AABB for '{target_prim_path}'.")
```

Not acceptable:

```python
if bounds is None:
    target_prim_path = f"{target_prim_path}/link_1"
```

Acceptable:

```python
if response.image.get("depth") is None:
    raise RuntimeError(f"Camera '{response.source}' has no depth frame.")
```

Not acceptable:

```python
depth = response.image.get("depth") or np.zeros((480, 640))
```

## Test Expectations

Add pure Python tests for:

- SDK registry and service resolution.
- Command builder and command feedback compatibility.
- Lease and power lifecycle.
- Manipulation command submission, feedback, failure, and override behavior.
- AO-Grasp request/response validation.
- CuRobo helper logic that can run outside Isaac Sim.

Add IsaacLab smoke tests for:

- Scene construction with required assets.
- Locomotion policy execution.
- Manipulation startup, AO-Grasp healthcheck, camera warm-up, target AABB resolution, planning, execution, and terminal feedback.

## Documentation Expectations

When changing architecture or workflow, update:

| File | When to update |
| --- | --- |
| `README.md` | User-facing setup, quick start, or major capability changes. |
| `ARCHITECTURE.md` | Pipeline, API, backend, or data-flow changes. |
| `PLANS.md` | Milestones, completed work, or known gaps. |
| Local directory README | Large directory ownership or launch commands change. |

## What Agents Must Not Do

- Do not add hidden fallbacks to make tests pass.
- Do not fake service responses in benchmark smoke scripts.
- Do not make target selection ambiguous.
- Do not bury critical config in hard-coded script globals when it belongs in YAML.
- Do not mix vendor code changes with benchmark integration changes unless explicitly requested.
- Do not commit generated `__pycache__`, temporary outputs, model downloads, or large debug artifacts.
- Do not weaken tests because the environment lacks Isaac Sim. Mark the limitation clearly instead.
