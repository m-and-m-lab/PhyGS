# Plans

This file tracks project status, milestones, and implementation priorities for the Interactive Search robotics benchmark.

## North Star

Build a repeatable IsaacLab benchmark where a Spot-style mobile manipulator searches for objects, localizes targets from onboard sensing, proposes grasps with AO-Grasp, plans arm motion with CuRobo, and reports task-level results through a Spot-SDK-shaped skills API.

## Completed

| Area | Status |
| --- | --- |
| Spot model assets | Spot arm USDs, camera-mounted USDs, URDF-derived assets, and CuRobo robot configuration are present. |
| Skills package | Core skill types, command builders, locomotion, manipulation, and Spot facade modules exist. |
| Spot command API | Mobility, arm, gripper command builders and feedback response objects are implemented. |
| SDK facade | `create_standard_sdk`, robot registry, service clients, lease, power, image, state, and manipulation clients are implemented. |
| IsaacLab SDK binder | Simulation-backed service backends bind the SDK facade to IsaacLab articulations, cameras, and manipulation runner. |
| Locomotion | Spot locomotion policy wrapper and WASD IsaacLab smoke test are implemented. |
| AO-Grasp client | HTTP client, healthcheck, pointscore request, proposal request, and response validation are implemented. |
| AO-Grasp Docker services | Pointscore and CGN sidecars have Dockerfiles and compose service definitions. |
| CuRobo integration | Joint command planner, stage collision world, depth collision, and local point-cloud ESDF planning are implemented. |
| Drawer smoke test | Fixed-base Spot, drawer, ground plane, cameras, AO-Grasp healthcheck, SDK lease/power, and manipulation feedback loop are implemented. |
| Unit tests | Command API, SDK API, manipulation API, AO-Grasp, CuRobo helper, locomotion API, and policy remap tests exist. |

## Current Milestone: Reliable Drawer Manipulation Benchmark

Goal:

Make the drawer manipulation smoke test a reliable benchmark entry point that can run repeatedly on a GPU IsaacLab machine and fail with actionable messages when dependencies are missing.

Tasks:

- Verify `tests/isaaclab_test/spot_manipulation_drawer.py` end-to-end in the IsaacLab Docker environment.
- Confirm AO-Grasp services are reachable from both host and container network modes.
- Confirm the whole drawer root AABB is visible from configured Spot cameras.
- Tune the default drawer pose only if planning or visibility consistently fails.
- Store run metadata such as target prim path, target center, selected proposal, accepted IK variant, feedback sequence, and terminal state.
- Add a minimal benchmark result JSON schema.
- Split GPU smoke tests from pure Python tests in CI or local test scripts.

Exit criteria:

- Drawer smoke test reaches `done` on a known machine with a documented command.
- Missing AO-Grasp sidecars, missing camera depth, missing target AABB, and CuRobo planning failure each produce clear non-zero failures.
- The README quick-start path matches the validated run commands.

## Next Milestone: Benchmark Metrics and Artifacts

Goal:

Turn smoke tests into measurable benchmark runs.

Tasks:

- Add a `BenchmarkRunResult` dataclass with task id, scene id, robot asset, target asset, timing, terminal state, and error message.
- Save debug artifacts under a deterministic output root, for example `outputs/benchmark/<timestamp>/<task_id>/`.
- Record AO-Grasp request id, point counts, proposal score, selected camera names, and selected grasp pose.
- Record CuRobo planning status, trajectory length, and collision-world update stats.
- Add success criteria beyond terminal feedback, such as gripper proximity, drawer joint displacement, or target contact.
- Add CLI flags for output directory, run id, and fail-on-metric thresholds.

Exit criteria:

- Each smoke test produces machine-readable results.
- A failed run contains enough information to reproduce and debug.

## Future Milestone: Mobile Manipulation

Goal:

Combine locomotion, perception, search, and manipulation into full interactive-search tasks.

Tasks:

- Define task configurations for starting pose, search target, target room/area, and success metric.
- Add a navigation/search skill that can command base motion through the same Spot SDK facade.
- Integrate locomotion and manipulation command arbitration.
- Add perception-based target proposal or query interfaces beyond a fixed target prim path.
- Support scene-level task manifests generated from Infinigen or manually curated USD assets.
- Add reset and episode lifecycle APIs.

Exit criteria:

- A single benchmark script can run an episode from initial navigation to final manipulation.
- All major robot actions are routed through SDK-shaped clients.

## Future Milestone: Task Library

Goal:

Expand beyond drawer manipulation into a suite of articulated and tabletop tasks.

Candidate tasks:

| Task | Assets | Success signal |
| --- | --- | --- |
| Drawer grasp/open | Drawer USDs | Drawer prismatic joint displacement and grasp completion. |
| Cabinet handle grasp | Cabinet articulated assets | Handle grasp and door/drawer motion. |
| Tabletop object pickup | Handheld objects and tables | Object lifted above threshold. |
| Microwave/oven handle | Articulated appliances | Door joint displacement. |
| Search in room | Infinigen indoor scenes | Target localized and approached. |

Implementation tasks:

- Add task YAML schema.
- Add target asset registry.
- Add per-task success evaluators.
- Add benchmark launcher that consumes task manifests.
- Add curated small scenes for deterministic regression testing.

## Future Milestone: Robust Developer Experience

Goal:

Make setup, testing, and debugging easier without weakening benchmark semantics.

Tasks:

- Add `Makefile` or task runner commands for pure tests, AO-Grasp services, IsaacLab smoke tests, and docs checks.
- Add environment validation script for CUDA, IsaacLab import, CuRobo import, camera assets, and AO-Grasp health.
- Add README snippets for common Docker network layouts.
- Add log summarization for long IsaacLab runs.
- Add lightweight type checks for pure Python modules.

## Known Gaps

| Gap | Impact |
| --- | --- |
| Full IsaacLab smoke tests are not runnable in lightweight Python-only environments | Runtime validation must happen in the IsaacLab container. |
| AO-Grasp model artifact paths depend on local Docker volume setup | New machines need clear model download or mount instructions. |
| Scene/task result schema is not standardized yet | Comparing benchmark runs is still manual. |
| Search and navigation are not yet task-level benchmark policies | Current benchmark is focused on locomotion smoke and manipulation smoke. |
| Vendor submodules are large and mixed with project code | Contributors must avoid accidental broad edits in `third_party`. |

## Decision Log

| Decision | Reason |
| --- | --- |
| Use a Spot-SDK-shaped API instead of direct runner calls | Keeps benchmark code close to real robot programming patterns and makes simulation backends swappable. |
| Fail fast instead of fallback-heavy launch scripts | Benchmark failures should expose missing dependencies and invalid assumptions immediately. |
| Target the whole drawer root prim in the drawer smoke test | Avoids hidden target narrowing and keeps target semantics explicit. |
| Use AO-Grasp sidecars over in-process services | Keeps heavy model dependencies isolated from IsaacLab runtime. |
| Keep IsaacLab binders under `skills.spot` | Separates public SDK-shaped API from simulation backend plumbing while keeping robot service binding discoverable. |
