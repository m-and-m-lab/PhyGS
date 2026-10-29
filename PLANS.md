# Plans

> PhyGS `develop`, 2026-10-28: this file came over from interactive-search with the code. Framework items apply as written (paths renamed, see CHANGELOG "Repository split"); benchmark items (tasks, tracks, episodes, results) belong to ForageBench and will move to its PLANS.md.


This file tracks project status, milestones, and implementation priorities for the Interactive Search robotics benchmark.
Status at a glance: [status board](docs/design/visual/status.html). Change history: [CHANGELOG.md](CHANGELOG.md).

## Robot portability (2026-10-28)

Design: [docs/design/robot_portability/index.html](docs/design/robot_portability/index.html); guide:
[docs/adding_a_robot.md](docs/adding_a_robot.md).

| Phase | Status |
| --- | --- |
| P0 guards | Done: `tests/test_robot_neutral.py` (no Spot names in framework code), a fake second robot in the unit tests (`tests/fixtures/robots/fake_mm`), the description check. |
| P1 description | Done: `RobotDescription` from `robots/<name>/robot.yaml`; joints, links, tools, cameras (roles) and the base pose come from it. |
| P2 package and plugins | Done: `robots/spot/`, `BaseController` (`legged_policy`, `planar_joints`, plugins), `RobotControl`, spec schema 2 (`robot.model`), `--robot`, `session.robot`. |
| P3 navigation and kinematics | Done: Nav2's motion model from the base kinematics, footprint from the profile, prismatic joints in IK. Differential bases refuse to navigate (the fine approach moves sideways). |
| P4 second robot | In bring-up: `robots/ridgeback_franka/` passes the description check, a simulator smoke (drive, arm, gripper, cameras), its measured footprint and the navigation course; the arm rungs with `--robot` are next (README there). |
| P5 differential base | Open. |

Spot's physical suite after the refactor: the robot-test ladder on the merged tree (see the robot
README's dated runs).

## Robot-test ladder (2026-10-28)

Every robot test in build order, 15 rungs ([phygs_robot_tests/README.md](tests/robot_tests/README.md);
`run_suite.py ladder --upto N`). What an agent asks for has agent tests (`episodes/`: a spec, a plan
through the session, checks from ground truth after and during the run via
`World.run(during=...)`); the machinery under the session has component tests
(`components/`, the robot built from the files episodes use). No seeds or success rates: those are
the benchmark's.

- Done: agent tests for navigation (course and house), obstacle avoidance, arm planning and pick and
  place (known boxes and depth, cuRobo and MoveIt) and the public entry point; the replaced component
  checks (navigation, dynamic obstacle, manipulation planning, house navigation) removed. Verified
  rung by rung: `output_check/20261028-0842_ladder_new_rungs`, `20261028-0910_ladder_moveit`,
  `20261028-0926_ladder_rerun_fixes`, `20261028-1038_merged_nav_search` (after the robot-portability
  merge). Full ladder on the merged tree: `20261028-1046_merged_full_ladder`, 22/23; the failure was
  the pick-place pre-grasp check measuring after the grasp (the closing jaw shifts the wrist 1.2-1.5
  cm). It now measures the pre-grasp before the line in (2-3.5 mm): `20261028-1424_pick_place_pregrasp_arrival`, 4/4.
- Open: search stays a component test (rung 13) until the episode harness has a search goal. The
  place error has a thin margin: 1.0-1.9 cm in 15 runs on 2026-10-28 against 2 cm (MoveIt the higher).

## Episode harness (2026-10-15)

Design and migration plan: [docs/design/episode_harness/index.html](docs/design/episode_harness/index.html).
One task (interactive search, retrieve: target held, arm stowed), one entry point
(`examples/run_world.py`), one oracle agent, a separate evaluator, flat robot tests.
Since 2026-10-18 the oracle is not an agent: its routines are in `tests/robot_tests/routines/`,
each episode robot test holds its plan, and the robot tests are in `episodes/` and `components/`
([robot tests housekeeping](docs/design/archive/robot_tests_housekeeping/index.html)).

- Step 1 done (2026-10-15): spec loader and the two grid specs, scene registry with the grid
  storage scene, `World` (main-thread loop, charged and realtime clocks, planner restart
  after the base moved, contacts, report and audit log), `AgentSession` with grant-checked hints,
  placeholder judge (since 2026-10-28 the retrieve evaluator, `tasks/retrieve.py`), template agent; unit tests (36). Verified headless:
  `output_check/20261015-1359_step1_template_table_b` (template agent: walk 0.6 m, stow; 8 s sim)
  and `output_check/20261015-1401_step1_table_pick_realtime_b` (a throwaway agent through the
  session: walk to the table stance, look, pick, stow holding; realtime clock, video; 33.6 s sim,
  one planner restart, no arm contact).
- Step 2 done (2026-10-15): the oracle (`agents/oracle.py`, strategy in `oracle.yaml`; since 2026-10-18
  `tests/robot_tests/routines/` and `routines/params.yaml`), built from
  `check_manipulation_planning.py`'s agent code with every check removed; robot tests
  `check_retrieve.py`, `check_drawer_stroke.py`, `check_drawer_place.py` (one episode per process,
  ground truth via `World.truth()`, thresholds in `spot_episodes.yaml`, now
  `tests/robot_tests/configs/episodes.yaml`). All pass headless
  with cuRobo: `output_check/20261015-1419_step2_retrieve_drawer` (70 s sim),
  `..._step2_drawer_stroke` (opened 0.335 m, closed 0.0003 m), `20261015-1531_step2b_retrieve_table`
  (45 s), `20261015-1531_step2b_drawer_place` (target in the drawer, upright, drawer closed; 156 s,
  video). The first drawer-place run stood still 0.081 m from the table approach stance for 250 s
  (tolerance 0.08 m, no navigation timeout): the oracle now gives each walk its own 60 s limit and
  issues it once more. The follower's stall near a goal was closed by the single-axis fine approach
  from settled poses (`execution/navigation/fine_approach.py`, 2026-10-18; 5/5) and the deployed
  policy's dead band (2026-10-22).
- Next: step 3, the house builder (read the house's bodies through `scene_assets.SceneBodies`, the
  scene-body collection of 2026-10-15), house annotations, one oracle parameter set for both
  scenes; then remove the episode code from `check_manipulation_planning.py`. Step 4: the UI on
  the World.
- Not wired yet: navigation or manipulation without `gt_localization` and `known_map` (refused
  at build); one World per process.

## Current manipulation milestone (2026-10-06)

Basic shared-runtime manipulation adds joint trajectories, arm gravity feedforward,
claw commands, free-space wrist differential IK, measured completion, cancellation,
and reset checks. Run `tests/robot_tests/components/check_manipulation.py --output DIR` for physical
validation; add `--record` for third-person video. See the tests README.

Done: MoveIt is qualified for collision-aware reaching with the full robot model,
and the reference grasp-and-lift passes 10/10. cuRobo V2 now passes the same full
check 10/10 (`output_check/cmp3_curobo_*`: pick, carry, refused carry, stop) after
fixing self-collision between links the arm never moves. Both planners sit behind one
contract (`planners/interface.py`). Planner refactor (2026-10-07): step 0 interface and
step 1 robot model are done (`output_check/model1_*`, 6/6): MoveIt checks the URDF meshes
plus USD camera boxes, cuRobo fits 244 spheres with `RobotBuilder`; plans take 0.03-0.3 s
(warm). Step 2 done (`output_check/ik2_*`, 4/4): skills use the libraries' IK (MoveIt KDL,
cuRobo IK solver) through `ik(IkRequest)`; our own IK was removed. Step 3 done
(`output_check/map3b_*`, 4/4, 0 N contact): each planner builds its own depth map with
its library through `observe(ObserveRequest)` (cuRobo `Mapper` ESDF, MoveIt Octomap); our
voxel-to-box export and cuRobo's cache sizing were removed; the 1 cm clearance is applied
by the libraries. Step 4 done (`output_check/grasp4c_*`, 4/4, 0 N contact): pick and place
use the libraries' grasp planning through `grasp(GraspRequest)` (cuRobo `plan_grasp`, MoveIt
Task Constructor); our yaw loop and approach check were removed. The planning model gained
the jaw's USD-only grip colliders. See
[Manipulation qualification update](#manipulation-qualification-update).
Done: the public `robot.manipulation.pick` skill (`docs/manipulation_skills.md`, step 1)
passes 10/10 through both map modes. Grasped objects are never attached in physics.
Step 2 done: a held object's box joins the planner request, the trajectory guard and
the depth map's robot voxels (never the physics); carrying passes 10/10.
Decision 2026-10-07: unknown space is not an obstacle in manipulation; the robot may
move into unseen space and the evaluator measures contact. Strict `require_observed`
and its map rules were removed from the robot test. In this mode pick and carry pass
10/10 (`output_check/gate_known_*`, `gate_depth_*`), 0 N contact.
Step 3 done: `place` sets the object on a different free table spot, 10/10 within
1.4–6.2 mm, upright, 0 N furniture contact (`output_check/place_gate_*`).
Step 4 done (2026-10-08): drawer `open`/`close` and the `pick` approach direction; gate
24/24 (`output_check/gate4`, both libraries, known boxes and depth, 0 N contact, a blocked
drawer stalls and lets go). Removed 2026-10-16: the `open`/`close` skills and `--task drawer`;
agents compose `move_along_axis` and the gripper commands (see `tests/robot_tests/routines/`; since
2026-10-18 `move_to_pose` and end-effector lines, `move_along_axis` removed).
Same day (2026-10-08): the robot USD's finger editor spheres were disabled, manipulation stillness uses a 0.05 s window, the planning check's guard uses 2 cm cells.
`stow` and `look_at` done (2026-10-09): gate 44/44 (`output_check/gate7`). Basic skills
and the house drawer done (2026-10-12): one attempt per skill, step narration, `stow` with a
held object, the house `corner_nightstand` top drawer; gate 42/44
(`output_check/20261009-1729_gate10`, `20261009-2210_rerun10`). Next: a full evaluated
episode in the grid scene, then the Bear house (needs the storage revision's per-body
solver iterations removed, and the robot's foot material bound). Grid episode done
(2026-10-12): `--task episode` (removed 2026-10-18; now the episode robot tests) passes with MoveIt and cuRobo. The walking policy's dead band is
handled in locomotion (`command_dead_band`, 2026-10-13); the grid episodes must be rerun with it.
The Bear house episode (`dev_house_drawer`: the small 9.5 x 9.5 x 7 cm cardboard box from the kitchen
island's bottom drawer, `retrieve_house`) is flaky (2026-10-23): it passed in
`output_check/20261022-1738_solver8_all` and `20261022-2229_episodes_9148873`, but the
`20261022-2159..2226_house_ab_*` A/B runs passed 1 of ~7 valid runs on HEAD and 0 of ~4 on the
working tree, failing as "object slipped while lifting" (the box against a ~13 cm open jaw gap),
"gripper closed on nothing" or a drop in the stow. Not a regression of the feedback and contact
change; the grasp geometry is the open item. The older AO-Grasp/SDK/CuRobo
stack was removed on 2026-10-08.
Walking policy replaced (2026-10-22): an arm-aware policy with a PhysX-drive stand, a feet-down
stand switch and a deceleration limit on stops (see `docs/locomotion.md`); the dead band is
0.08 m/s (0.08 rad/s) in every direction and the goal tolerance stays 4 cm. Navigation plans a
rectangle footprint (standing outline + 0.05 m) in position and heading (built-in; Nav2 plans
the enclosing circle, its rectangle mode tried and not robust with RPP yet). The phantom joint
velocity of the PhysX-drive stand (arm 6-8 mrad off target) was fixed by the solver at 8 TGS
iterations with gravity every iteration (2026-10-22; `docs/locomotion.md`). The Nav2 known-map and
dynamic detours pass after the RPP tuning and a 60 s budget (`20261022-1929_nav2_tuned`,
`20261022-1953/1956_nav2_dynamic_budget60_r1/r2`).

## North Star

Build a repeatable IsaacLab benchmark where a Spot-style mobile manipulator searches for objects, localizes targets from onboard sensing, grasps and manipulates them, and reports task-level results from independent physical evaluators. Agents act through small public skill APIs (`robot.locomotion`, `robot.navigation`, `robot.manipulation`) on one shared runtime. Grasp-proposal and planner libraries are replaceable backends, not fixed choices.

## Shared-framework implementation sequence

The navigation-first direction supersedes the older drawer-first milestone below.
The older sections remain an inventory of existing components and outstanding work.

1. **Implemented: initial lifecycle foundation.** Namespaced package, extracted Spot
   scene factory, single-environment state restoration, integer runtime clock,
   exclusive position-controller ownership, human authority and terminal estop.
   Unit checks and the real Spot lifecycle diagnostic are in `tests/`.
2. **Implemented: locomotion integration.** Preserved checkpoint preprocessing/timing,
   restricted policy writes to legs, added runtime-backed velocity/stop/feedback and
   arm posture clients, migrated keyboard control, and added real tracking/stopping/
   contact checks. Fixed PD reset targets and runtime-managed skill APIs.
3. **Implemented:** timestamped joint/IMU/RGB-D observations, calibrated capture-time
   transforms, explicit ground-truth/no-localization configurations, permission
   filtering, reset isolation, and one visual/automated observation diagnostic.
4. **Implemented:** known-box-map point navigation, A* and local tracking,
   independent physical evaluation, permitted action interfaces, manual goal
   selection and automated navigation/direct-motion agents.
5. **Implemented, partially validated:** stable navigation backend contract, optional Nav2 adapter,
   measured folded-arm/leg transit envelope and paired physical-report comparison. Built-in
   simulation and calibration passed; Jazzy/Nav2 imports and the live eight-case
   Nav2 run pass in IsaacLab Python. The paired backend report comparison is available.
6. **Implemented and physically checked: depth mapping helper.** Capture-time depth rays,
   bounded 3D occupancy evidence and conservative free/occupied/unknown projection;
   one manual/automated mapping diagnostic. Uses declared ground-truth localization.
7. **Implemented for a declared flat-floor diagnostic:** an observed-depth map provider
   blocks unknown space and connects to the shared navigation API. Both backends now
   accept its evidence; see `docs/mapping.md` for the Nav2 launch configuration. A verified free
   spawn region plus a turning depth scan supports one physically successful
   point-navigation route; probe-height free evidence assumes tall obstacles.
   General floor support, estimated localization and frontier selection remain
   future work, before manipulation and interactive search.

8. **Implemented: first shared-runtime object-search task.** A reference agent uses
   the existing navigation API and raw RGB-D observations, then submits a stamped
   image pixel. An independent evaluator checks physical target bounds and measured
   depth, including occlusion and cross-episode evidence. The Isaac Sim diagnostic
   passes for one occluded cuboid scene; broader assets, learned detectors and
   frontier search remain future work.

9. **Implemented: optional observation/processing/navigation pipelines.** Agent and
   service inputs are independently filtered under task-wide permissions, including
   transforms and indirect privileged assistance. Runtime-managed processing is
   optional; navigation factories own their dependencies. A sensor-only localization
   estimator interface and explicit goal frames support external models without a
   universal map representation. See `docs/pipelines.md` for inspection commands.

10. **In progress: manipulation.** Basic joint/wrist/gripper commands in the shared
    runtime are implemented and checked. MoveIt collision-aware reaching and
    grasp-and-lift are qualified in the table scene. The skill-level API is approved in
    `docs/manipulation_skills.md`; `pick`, `place`, `look_at` and `stow` are implemented
    and gated (2026-10-12: 42/44), `move_along_axis` since 2026-10-13. The `open`/`close`
    skills were removed on 2026-10-16, and `pick`, `place` and `move_along_axis` on 2026-10-18
    (manipulation contract, `docs/design/manipulation_contract/`): agents compose them from
    `move_to_pose`, end-effector trajectories and the gripper. See the qualification update below.

The shared foundation has a point-navigation task specification. General task
composition, a physically validated estimated-localization provider, controller actuator-mode switching and
scene-wide asset inventory validation remain future work. See `tests/README.md` for exact checks and commands.

## Removed stack

The Spot SDK facade, AO-Grasp client and sidecars, CuRobo runner, camera/point-cloud
helpers, the old main launcher and the fixed-base drawer smoke tests were removed on
2026-10-08. See [ARCHITECTURE.md](ARCHITECTURE.md#removed-spot-sdk-facade-ao-grasp-and-curobo-runner).

## Backlog (pre-runtime milestones)

The milestones below were written for the removed SDK/AO-Grasp stack, before the
shared-framework sequence. That sequence supersedes the drawer-first milestone.
They remain as a backlog; reuse the task and metric ideas on the shared runtime.

### Superseded: Reliable Drawer Manipulation Benchmark

Goal:

Make the drawer manipulation smoke test a reliable benchmark entry point that can run repeatedly on a GPU IsaacLab machine and fail with actionable messages when dependencies are missing.

Tasks:

- (Removed 2026-10-08) The drawer smoke test itself; drawers are checked by
  `tests/robot_tests/episodes/check_drawer_stroke.py` and the other episode tests on the shared runtime.
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

### Benchmark Metrics and Artifacts

Goal:

Turn smoke tests into measurable benchmark runs.

Tasks:

- Add a `BenchmarkRunResult` dataclass with task id, scene id, robot asset, target asset, timing, terminal state, and error message.
- Save debug artifacts under a deterministic output root, for example `outputs/benchmark/<timestamp>/<task_id>/`.
- Record grasp planner requests, the chosen candidate, and the selected grasp pose.
- Record CuRobo planning status, trajectory length, and collision-world update stats.
- Add success criteria beyond terminal feedback, such as gripper proximity, drawer joint displacement, or target contact.
- Add CLI flags for output directory, run id, and fail-on-metric thresholds.

Exit criteria:

- Each smoke test produces machine-readable results.
- A failed run contains enough information to reproduce and debug.

### Mobile Manipulation

Goal:

Combine locomotion, perception, search, and manipulation into full interactive-search tasks.

Tasks:

- Define task configurations for starting pose, search target, target room/area, and success metric.
- Add a navigation/search skill that commands base motion through the shared runtime (done: `robot.navigation`).
- Integrate locomotion and manipulation command arbitration.
- Add perception-based target proposal or query interfaces beyond a fixed target prim path.
- Support scene-level task manifests generated from Infinigen or manually curated USD assets.
- Add reset and episode lifecycle APIs.

Exit criteria:

- A single benchmark script can run an episode from initial navigation to final manipulation.
- All major robot actions are routed through the skill clients.

### Task Library

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

### Robust Developer Experience

Goal:

Make setup, testing, and debugging easier without weakening benchmark semantics.

Tasks:

- Add `Makefile` or task runner commands for pure tests, IsaacLab robot checks, and docs checks.
- Add environment validation script for CUDA, IsaacLab import, MoveIt and cuRobo imports, and camera assets.
- Add README snippets for common Docker network layouts.
- Add log summarization for long IsaacLab runs.
- Add lightweight type checks for pure Python modules.

## Known Gaps

| Gap | Impact |
| --- | --- |
| Full IsaacLab smoke tests are not runnable in lightweight Python-only environments | Runtime validation must happen in the IsaacLab container. |
| Scene/task result schema is not standardized yet | Comparing benchmark runs is still manual. |
| General search assets and policies are not yet supported | The first shared-runtime object-search task validates RGB-D evidence in a diagnostic box scene; broader detection, scene layouts and frontier strategies remain future work. |
| Wall-clock simulator performance remains below real time | Recorded two-camera timings attribute the largest costs to actuator/scene writes (35%), RGB-D capture (29%), and physics stepping (21%); policy computation is about 2%. Removing per-tick RGB-D copies improved throughput, but 250 Hz physics plus 31.25 Hz RTX capture still needs optimization before 30 wall-clock FPS can be expected. |
| Vendor submodules are large and mixed with project code | Contributors must avoid accidental broad edits in `third_party`. |

## Decision Log

| Decision | Reason |
| --- | --- |
| Navigation-first shared runtime with `robot.<skill>` clients (supersedes the drawer-first plan) | One clock, joint ownership and evaluator for every skill; agents never step physics. See the implementation sequence above. |
| Use a Spot-SDK-shaped API instead of direct runner calls (superseded; the stack was removed 2026-10-08) | Kept benchmark code close to real robot programming patterns; the skill clients now keep Spot SDK vocabulary. |
| Fail fast instead of fallback-heavy launch scripts | Benchmark failures should expose missing dependencies and invalid assumptions immediately. |
| Target the whole drawer root prim in the drawer smoke test | Avoids hidden target narrowing and keeps target semantics explicit. |
| Use AO-Grasp sidecars over in-process services (superseded; removed 2026-10-08) | Kept heavy model dependencies isolated from the IsaacLab runtime; planners now run in a separate worker process. |
| Keep IsaacLab binders under `skills.spot` (superseded; removed 2026-10-08) | Separated the SDK-shaped API from simulation plumbing. |

## Progress notes

### API/execution package layout

Public locomotion/navigation APIs live in `phygs/skills/<name>/api.py`.
Implementations live under `phygs/execution`; observation helpers
remain under `helpers/observations`. Spot's policy, checkpoint and configuration files live in its
robot folder (`robots/spot/`). Robot assembly binds the public clients to the runtime.
Manipulation execution lives in `phygs/execution/manipulation`.

- Added a dedicated `check_nav2_pipeline.py` robot-test launcher for real Nav2
  planner/controller processes, reusing known-map and observed-map physical checks.
  Native Nav2 point-cloud costmaps are now covered separately; sensor-based
  localization and BT orchestration remain future work.

- Physically validated active-route invalidation for the Nav2 point-cloud adapter.
  The moving-obstacle diagnostic checks Nav2 costmap updates, replan, positive
  physical clearance, controlled stopping, and costmap reset isolation.
- Added a separate Nav2 `PointCloud2` obstacle-layer configuration. It uses Spot's
  timestamped depth and calibrated optical transform with a declared planar-floor
  prior. Ground-truth localization remains the initial integration mode.

### Bear-house navigation and exploration integration (2026-10-02)

Added the shared house diagnostic and native Nav2 launcher selection, explicit
640×480 / 0.032 simulated-second camera configuration, USD body registration for
reset, and an optional frontier-viewpoint selector using observed maps. The real
Nav2 house navigation check passed two episodes: physical arrival, blocked-goal
rejection, cancellation, stopping and exact restoration of 229 scene bodies.
Camera delivery measured 31.25 simulated FPS, with RTF 0.18–0.20. The bounded kitchen search check also passed: a depth-derived frontier goal,
physical locomotion, grounded target declaration, cancellation and stopping. It
measured 31.25 simulated camera FPS at RTF 0.145. The independent coverage-only
case also passed: autonomous viewpoint navigation, scan and grounded declaration,
with 31.25 simulated camera FPS at RTF 0.139. Unrestricted whole-house search and 5 cm terminal
positioning remain unvalidated; the latter failed a physical test. FPS optimization remains deferred. Ground-truth
localization is the supplied configuration; agent-owned localization remains an
external integration choice.


## Manipulation qualification update

Known-map reaching with the full robot model is qualified with MoveIt. cuRobo
still needs full-model qualification. Shared multi-camera mapping and conservative
hull pruning remain supported. The planning check now uses only onboard cameras;
the virtual views and robot self-filter were removed. With a declared hand-camera
look step, MoveIt observed-obstacle depth reaching passed (`onboard_depth_look_01`):
0.81 mm wrist error, 0 N contact. The floating-box scenario was replaced by a table
scan-and-reach task (5 layouts, ground-truth target pose, onboard cameras, 4-pose
hand-camera scan). Voxels the robot body fills during the episode count as free.
2026-10-05 comparison (`output_check/cmp_*`): MoveIt known 5/5, strict 2/5 (left,
right, near fail on scan coverage at arm height and the 30 s budget; strict plans
take 60–112 s with ~850 voxel boxes). cuRobo V2 0/10: with the full-robot model it
rejects even a zero-length plan with no obstacles and no self-collision (cost 1e16).
2026-10-06: a six-pose scan chosen by rendered coverage (74% -> 100% of the space
MoveIt's arm sweeps) and exact merged voxel boxes (~848 -> ~270) make MoveIt strict
pass 5/5 layouts (`output_check/scan6_*`): plans 11–20 s, wrist error 1.0–3.3 mm,
0 N contact. cuRobo still needs a separate model/config investigation.
Grasp and lift (`output_check/final_grasp_moveit_*`): MoveIt passes 10/10 (5 layouts,
known and strict), target lifted 18.3–18.5 cm, 0 N arm contact with furniture. The
gripper now holds a bounded squeeze after stopping on an object, judged from measured
positions; the reference agent picks a grasp yaw whose open-gripper descent is clear.
Skill interface approved (`docs/manipulation_skills.md`). Step 1 done: the public
`robot.manipulation.pick` skill with `collision_avoidance_planner` and `direct` backends
passes 10/10 MoveIt runs (`output_check/skill_known_*`, `skill_strict_*`): lift
18.3–18.6 cm, 0 N furniture contact, 1 attempt each. `direct` passes `center`, but its
open finger hits the clutter on `near` (10.7 N, `skill_direct_near/collision_frames.png`).
Grasped objects are never attached in physics.
Step 2 done (`output_check/carry3_known_*`, `carry5_strict_*`): over the clutter, a goal
only the held object would hit is refused (wrist moves 2.3–3.4 mm), a goal 3 cm higher
succeeds, the arm returns, and held-object and arm contact stay 0 N, 10/10. Without the
held object in the plan request, MoveIt plans the colliding carry and the guard stops it
(`carry_without_attachment_01`). Fixed on the way: the planner now uses the measured
gripper angle; arm-only commands no longer reset the gripper squeeze (an object dropped
when a pose command started); planner cleanup no longer hides errors; the depth map keeps
the robot's current voxels free inside contact regions. Open: while holding, the wrist
pitch joint sits 0.018 rad below its target (cause not explained). Next: `place`, drawer
`open`/`close`, and a full episode.

### Earlier manipulation notes (superseded by the 2026-10-05/06 results above)

Strict depth-based reaching is blocked by the robot-volume/unknown-space contract.
Experimental visibility and containment probes have been removed. The next gate
is one physically validated depth-based reach with explicit self-collision,
initial-state validity, and newly entered unknown-space rules. Grasping and
articulated-object interaction follow that gate. See
[the current limitation](docs/manipulation_planners.md#depth-collision-policy).

Observed-obstacle depth reaching now has one passing MoveIt physical experiment
(`depth_reach_04`). Strict observed-space certification remains blocked. Next,
address the measured map/validation wall-time cost and repeat this same reach
across declared poses before extending to grasping. Avoid new diagnostic layers.
