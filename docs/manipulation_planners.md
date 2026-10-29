# Collision-aware manipulation experiments

Two real backends use the same Spot model, observation packet, known boxes or
depth images, and runtime executor: MoveIt 2 with OMPL RRTConnect, and cuRobo V2. Neither backend
steps physics or writes actuator targets. These are **reaching experiments**, not
a grasping or articulated-object opening implementation.

**Changed 2026-10-13.** The independent trajectory guard, its hook on `JointTrajectory` and
`execution/manipulation/validation.py` were removed: nothing checks or stops a planned motion
while it runs; the evaluator records contact and judges the outcome. Skills solve IK with
`execution/manipulation/kinematics.py` (multi-start least squares, joint limits only) instead
of the planners' IK (MoveIt's single-seed KDL with 0.05 s and cuRobo's inflated self-collision
model both refused poses the arm reaches). Mentions of the guard below are history.

## Run

From the repository root, with ROS Jazzy sourced (MoveIt only):

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=47
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_arm_planning.py \
  --spec scripts/PhyGS/tests/robot_tests/configs/episodes/table_reach.yaml --planner moveit --record \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_arm_planning_moveit
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_pick_place.py \
  --spec scripts/PhyGS/tests/robot_tests/configs/episodes/table_reach.yaml --planner moveit --record \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_pick_place_moveit
```

These are rungs 8–10 of the [robot-test ladder](../tests/robot_tests/README.md): agent tests whose
plan drives the arm through the agent session at the grid table (a target next to one clutter box).
The run is automatic and headless; add `--viz kit` to watch. `--planner curobo` (the default) uses
the other planner. The spec decides what the planner avoids: the scene's known boxes
(`table_reach.yaml`) or the onboard depth cameras (`table_reach_depth.yaml`). House and grid episodes
are covered by the other agent tests (`check_retrieve.py` and `check_drawer_stroke.py` on
`dev_grid_drawer.yaml` and `dev_house_drawer.yaml`, `check_drawer_place.py` on `dev_grid_table.yaml`).
Use a fresh output directory for every run.

`check_arm_planning.py` looks at the target, moves the hand over the clutter and over a free spot,
cancels a planned move, stows and walks with the cameras and the planner resident; it also checks
that the planner's robot model sits on the physical links and that it refuses a start inside an
obstacle. `check_pick_place.py` picks the target with the routines' pick (a planned `move_to_pose` to
the pre-grasp, a straight end-effector line in, close, a line out), refuses a carry goal only the
held object would hit (known boxes only: on depth the clutter's top is unseen while holding, see
below), carries it over the clutter and back, places it on a free spot and stows. Both read the
target pose and every arm contact on the test side (ground truth, never the agent's). Outputs:
`check.json`, the episode's `report.json` and `audit.jsonl`, the planner's log and requests, and
`third_person.mp4` with `--record` (simulation time).

## Dependencies

MoveIt runs in system Python; cuRobo runs in a separate Isaac Python process.
This avoids mixing MoveIt's binary Python dependencies into the simulator.
Neither is a required dependency of the basic manipulation API. The container
(`docker/Dockerfile.phygs`) installs both; outside it, the same steps are:

```bash
apt-get install -y --no-install-recommends \
  ros-jazzy-moveit-py ros-jazzy-moveit-planners-ompl
```

The planners are qualified with MoveIt 2.12.4 and cuRobo V2 commit
`78fd485fa82d9b9a063fb4985e371814587e666a`. V1 configuration/API is different.
cuRobo V2 goes in its own folder, which `planner.curobo_dependencies` in
`phygs/configs/environment.yaml` names (`/opt/curobo_deps`):

```bash
git clone https://github.com/NVlabs/curobo.git /tmp/curobo
git -C /tmp/curobo checkout 78fd485fa82d9b9a063fb4985e371814587e666a
./isaaclab.sh -p -m pip install --target /opt/curobo_deps --no-deps \
  /tmp/curobo yourdfpy==0.0.60 numpy-quaternion==2024.0.13 \
  setuptools_scm==10.3.4 'cuda-core[cu12]==1.2.1'
```

This reuses the simulator's existing Torch 2.10/CUDA 12.8 and Warp 1.13. A missing
folder stops the run with an error; there is no fallback.

## Where the code lives

| Location | Responsibility |
|---|---|
| `phygs/skills/manipulation/api.py` | Agent commands; optional configured pose planner; trajectory admission |
| `phygs/execution/manipulation/planning.py` | Named, timed trajectory contract and isolated worker transport |
| `phygs/execution/manipulation/kinematics.py` | Arm forward and inverse kinematics from the planning URDF (joint limits only) |
| `phygs/execution/manipulation/planners/` | Vendor adapters, worker entry point, shared Spot planning model |
| `phygs/execution/manipulation/planners/interface.py` | Common planner contract: `plan(PlanRequest)` (depth images included) and `ik(IkRequest)`. MoveIt (KDL `set_from_ik`, Octomap) and cuRobo (`ik_solver.solve_pose`, `Mapper`, `RobotSegmenter`) implement it. The skills plan to a start pose with it, then move straight. |
| `phygs/execution/manipulation/arm_planner.py` | `ArmPlanner`: the one planner host for episodes (and their robot tests) and the UI; known boxes, the robot's depth cameras, or both |
| `phygs/execution/manipulation/controller.py` | Simulation-time reference following and physical completion |
| `phygs/execution/manipulation/collision_map.py` | Known boxes into the body frame |
| `tests/robot_tests/episodes/check_arm_planning.py`, `check_pick_place.py` | Agent tests: a plan through the session, independent physical assertions |
| `tests/robot_tests/configs/arm_planning.yaml`, `pick_place.yaml`, `episodes/table_reach*.yaml` | Thresholds; the table scene and what the planner avoids |

The episode harness hosts the planner (`ArmPlanner`) and the agent calls
`session.manipulation.move_to_pose(...)`. A planner failure propagates; it never silently switches
to the free-space IK controller.

## Shared observations and collision geometry

Depth comes from Spot's onboard Orbbec Gemini 2 cameras (`frontleft`, `frontright`,
`hand`) at 640×480 and 31.25 simulated Hz. Depth nearer than 0.2 m is no data
(`ObservationConfig.depth_min`), as on the real sensor; the renderer draws surfaces from
5 mm, so nothing is seen through (2026-10-25). A held box therefore hides what lies behind
it from the hand camera. The same observation packet carries
image calibration, capture-time transforms and joint measurements. Ground-truth
localization is explicitly selected.

Each plan builds a new depth map (decision 2026-10-23): the arm moves while the base
stands, so a map in the body frame, rebuilt from the current images, holds no stale
obstacles (a picked box, a moved drawer) and needs no clearing rules. `ArmPlanner`
waits for the first image of every camera taken after the request (the arm is still),
sets pixels outside the body-frame work volume to 0, and sends the images with the
`PlanRequest`. The worker clears its map, fuses the images and plans: cuRobo fuses every
pixel with its `Mapper` (TSDF with a 4 cm truncation band, then an ESDF at 2 cm; with one
image per camera a 2 cm band left gaps in a surface seen at a grazing angle); MoveIt fuses every 4th pixel per
axis into its Octomap (2 cm) through the point-cloud updater. Each library removes the
robot's and the held object's pixels itself, from the measured joints only (no world
pose): cuRobo's `RobotSegmenter` on the planner's own collision spheres (legs, gripper
and attached held object), MoveIt's Octomap shape mask on the link frames, which the
worker publishes in TF (`body` to each link at the request's joints) with each image,
and on the attached held object. `self_filter_padding` (5 cm, cuRobo's default) grows both
masks: at 2 cm, about 40 silhouette pixels of the gripper seen by the front cameras stayed in
the map and made a stow start in collision (episode `retrieve_grid` without `known_map`,
2026-10-24; replayed offline, 3 cm already planned). Scene depth that close to the robot is
not mapped.

The map holds only what the cameras see at that moment. While the robot holds an object, the
held object hides the view ahead from the hand camera, and the front cameras sit below the
table's clutter: its top is then unseen and therefore free. The planning test checks the
"held object would hit the clutter" refusal on known boxes only (decision 2026-10-25; it passed
on depth before only because the renderer showed through the held box). The front cameras look at the
floor near the feet, so the hand camera gives most of the view: an agent looks at the
work area before it plans (in depth runs the test's pick looks at the target first).
Unseen space is free (see [depth collision policy](#depth-collision-policy)).

Our own depth mapper still owns free/occupied/unknown evidence for navigation (2D
projection) and for the independent trajectory guard (3D voxels, 2 cm, as fine as the
planners' maps). Planner and guard therefore use different maps; the guard stays the
referee. The
known-map guard checks arm-floor clearance separately from the legs' intentional
support contact.

Clearance: both planners keep `obstacle_padding` (1 cm) between the robot, the held
object and every obstacle, through the library. MoveIt pads the robot links and the
attached object (`default_robot_padding`, `default_attached_padding`; its self-collision
check uses the unpadded model). cuRobo grows the boxes and lowers the ESDF by 1 cm.
cuRobo's sphere buffer (`collision_sphere_buffer`) was not used: with it, the start
state self-collided by 0.47 mm.

The planners and the referee use different robot geometry on purpose:

- **Planning URDF** (`prepare_model`): the URDF collision meshes plus one box per
  Gemini camera housing and per USD-only collider (the jaw's grip spheres), measured from
  the USD (`usd_only_boxes`; the URDF has neither).
  MoveIt checks the static links (body, legs) as these shapes, and the arm links as the
  referee's spheres (`spot_planning_moveit.urdf`, 2026-10-09). The referee's spheres reach
  up to about 5 cm past the arm's collision meshes, more than the 1 cm clearance: MoveIt
  paths that passed close over the open drawer's front were rejected by the guard. The stow posture self-collides in these
  spheres, as in cuRobo's; the pre-stow does not, and no recorded gate7 waypoint does.
  MoveIt's SRDF also skips sphere-modelled arm links two joints apart (el0–sh0, el0–wr0,
  el1–sh1, el1–wr1, fngr–wr0): their spheres overlap around the joint between them, e.g.
  wrist and forearm after a drawer release, where the meshes do not touch.
  cuRobo fits its own spheres to them with
  `RobotBuilder` (342 spheres, 77 on the jaw; 244 before the jaw boxes) and computes its collision
  matrix, adding our skip list for adjacent links and links the arm does not move. The fit
  is cached by URDF content in `~/.cache/phygs/curobo` (about 37 s cold, 5 s
  cached).
- **Referee** (trajectory guard, map robot space, approach check): conservative spheres
  from the USD collider bounds, including hidden `guide` colliders, legs, feet and camera
  hardware, every grid cell enclosed by a sphere. The report includes its coverage audit.

Non-arm joints are locked at the measured configuration for planning; this is not a
whole-body planner.

## Execution qualification

Planning requests use a worker future. Physics, locomotion and cameras continue
while a command holds its measured arm position. Cancellation or episode reset
discards pending results. Completed plans are checked against current joints and
base pose before execution. Startup remains synchronous.

The controller stops on excessive tracking error; stopping retains ownership until measured
settling. (Until 2026-10-13 a trajectory guard also re-checked intermediate configurations
against the current collision evidence; it was removed.)

The shared planner input adds an explicit 1 cm obstacle margin; independent
validation uses the actual obstacle dimensions. This margin is not a formal
tracking-error or stopping-distance certificate.

The MoveIt configuration now uses finer OMPL segment checking and zero TOTG path
blending tolerance. An earlier timed trajectory clipped the obstacle between clear
waypoints; the independent guard rejected it before execution. The larger geometry
also needs a larger explicit planning budget. Historical performance results below
are for the earlier model and must not be compared directly with current timings.

Looking first improves what the planner knows; it is not required before moving (see
[depth collision policy](#depth-collision-policy)). Automatic view selection is not
implemented.

## Grasp and lift

After the pre-grasp reach, the reference agent grasps the ground-truth target top-down:
open, descend 0.19 m with `move_joints` (its own inverse kinematics), close, lift back to
pre-grasp, hold 1 s, place, release, retreat and close the empty gripper. The target is a
free 0.2 kg, 4 × 4 × 12 cm box; the table, clutter and probe are fixed. Pass criteria:
the target rises at least 5 cm while held, and no arm link touches the table, clutter or
probe (filtered contact sensor, 0.5 N limit). Contact between the gripper and the target
is intended and not counted. The descent depth comes from the USD gripper colliders:
the fixed jaw spans 0.146–0.236 m along the wrist axis.

Removed 2026-10-23: `grasp(GraspRequest)`, MoveIt's Task Constructor task and cuRobo's
`plan_grasp` call. Since 2026-10-13 the skills plan to a start pose (`plan`) and move straight
from there, and nothing called `grasp` any more. The record below is kept as history.

Grasp planning (step 4, 2026-10-07): the skill sends one `GraspRequest` with the four
top-down yaws (0, −90, 90, 180 degrees) as candidates. The library picks one and plans
the approach, the straight descent and the straight lift with the gripper open and full
collision checking. cuRobo: `plan_grasp` with a goal set and no disabled links. MoveIt:
a Task Constructor task in the worker (start from the live planning scene, OMPL
`Connect`, one alternative per candidate with a Cartesian descent and IK, Cartesian lift).
The approach `Connect` gets `planning_time_seconds` split over its at most 4 IK goals per
candidate (`IK_SOLUTIONS`); MTC's default of 1 s per approach plan timed out on the
handle beside the open drawer once MoveIt checked the arm as the referee's spheres.
MoveIt times paths at 1.5 rad/s and 1.5 rad/s² (no jerk limit): at 3.0 rad/s² a swing
from a high view down to a drawer handle reached 4.5 rad/s² on `arm_wr0` (cuRobo's
jerk-limited profile of the same path: 2.4) and swayed the legged base past the guard's
0.03 rad.
The object to pick is left out of the boxes, or its pixels out of the depth images; the
planner map is rebuilt from the stored images without it. Only the approach goes through
the independent guard, because the guard's map still holds the target.

Gate (`output_check/grasp4c_*`, 4/4, 0 N contact): pick, carry, place. Grasp plans:
MoveIt 0.5-0.6 s, cuRobo 4.6-4.7 s (the first call builds the open-gripper planner;
0.6 s warm offline). Place plans: MoveIt 0.15 s, cuRobo 0.6-1.0 s. Place error 1-3 mm.

Two causes found on the way:

- The jaw has 10 grip spheres only in the USD; the planning URDF lacked them, so the real
  jaw was ~4 cm longer than the planned one and hit the clutter with 7.3 N on `near`
  (`output_check/grasp4b_moveit_known_near_contact`). `usd_only_boxes` now adds a box per
  USD-only collider, as for the camera housings.
- MoveIt Task Constructor: the worker keeps one `PipelinePlanner` for its lifetime;
  destroying the last one unloads the OMPL plugin that MoveItPy also uses, and its next
  plan segfaults.

After the pick, the test checks carrying with the held object in collision checking
(`docs/manipulation_skills.md#held-objects-in-collision-checking`). Over the clutter
box, a goal that leaves the gripper 1.5 cm above it but sinks the hanging object into it
must be refused, and the wrist must move < 2 cm. A goal 3 cm higher must succeed, then
the arm returns. A filtered contact sensor on the target fails the run on any contact
between the held object and the table, clutter or probe during the carry. Without the
held object in the plan request, MoveIt plans the colliding carry; the independent guard
still stops it before contact (`output_check/carry_without_attachment_01`).

The depth map treats the voxels where the robot and its held object are now as free;
cameras see their surfaces. Voxels where a camera saw the held object earlier can stay
occupied after release; this makes the planner more careful, never less safe.

Wrist offset while holding: the wrist pitch joint sits 0.018 rad below its target with
the 0.05 rad grip squeeze, and 0.003 rad with a 0.01 rad squeeze; without an object it
is 0.0001 rad. The offset scales with the squeeze, so the gripper drive's squeeze leaks
into the wrist. Measured 2026-10-17 (`output_check/20261017-0032_grip_leak`, a free box in
the jaws, the arm raised): the leak is a PhysX (TGS) artifact, as the squeeze is internal to
the hand. It grows with the squeeze torque times the physics step, up to ~0.042 rad on
`arm_wr0`; it does not depend on the object's weight, the depenetration speed or the actuator
grouping, and PGS gives 0.026 instead of 0.042. Halving the gripper drive (60 / 10 instead of the
USD's 120 / 20) did not reduce it at 250 Hz, where the cap is already reached, and the robot then fell stopping from 1 m/s; the drive stays 120 / 20
(`output_check/20261017-1443_stop_fall_crosscheck`). `solve_articulation_contact_last` removes the
leak (+0.004 rad) but the arm then lags its references (0.15 rad) and Spot fell in the
manipulation checks (`output_check/20261017-1033_contact_last_eval`). It is about 5 mm at
the fingertips; placement still lands within 2 mm in the place check. After a successful
motion the arm keeps its commanded targets (on release, while a plan is computed, and as the
start of the next command), so the offset does not add up; re-targeting the measured position
let a held load push the arm once more each time (0.034 rad on `arm_el1` during a 5 s stow
planning wait at 250 Hz, refused by the trajectory start check, 2026-10-16). A failed or
cancelled motion holds where the arm stopped. `grip_squeeze`
must exceed `joint_tolerance` (0.03 rad), or holding cannot be detected; the motion
configuration rejects smaller values.

The planner locks the gripper closed except for the measured angle sent with a held
object, so the agent closes the empty gripper before planning again. Following common
pick pipelines (pre-grasp, straight approach, close, lift), contact with the target is
allowed only in the final phase.

## Skill design notes (survey, 2026-10-06)

A survey of Spot SDK, MoveIt Task Constructor, BEHAVIOR/OmniGibson, Habitat/HomeRobot,
PARTNR, AI2-THOR, ManiSkill and LLM-robot frameworks found:

- Agent APIs take object ids or names: `pick`, `place`, `open`, `close`, `release`,
  plus a low-level pose command. Pixels, rays and poses live in robot SDKs and planners.
- Collision avoidance is split: planner-based systems avoid the scene inside the skill;
  the real Spot arm avoids only itself and its body, so our planner is a benchmark layer.
- Where planning exists, the target is an obstacle in transit and exempt only for the
  final straight approach; the held object is then attached to the robot model.
- Doors and drawers: grasp the handle, then a force-limited motion along an axis
  (Spot `ConstrainedManipulationCommand`, reporting stuck or grasp lost).
- Feedback is a status plus a reason (precondition, no grasp, planning, execution,
  postcondition) and often a contact-force budget in evaluation.

The current pick sequence follows this pattern in the reference agent. Object-level
skills, attached-object planning and opening are not implemented yet.
Note 2026-10-16: there is no `open`/`close` skill (removed). Since 2026-10-18 there is no
`pick`, `place` or `move_along_axis` either: agents compose them from `move_to_pose`,
end-effector trajectories and the gripper commands (see the robot tests' routines,
`tests/robot_tests/routines/`).

## Limits that matter

- RTX render hang with cuRobo on the same GPU (2026-10-08/09): 4 of about 70 cuRobo robot
  runs stopped for good, none of the MoveIt runs. Thread dumps (`stacks.txt`, written by
  `faulthandler` every 15 wall minutes) show the main thread inside Isaac Sim's camera render
  (`SyntheticData._post_process_graph_tick` from `Camera.data`) while the cuRobo worker
  computes a grasp plan on the same GPU. Not proven to be the cause; the next test is to run
  the cuRobo worker on a different GPU than rendering. A rerun of the same case passes.

- Unknown space is not an obstacle (see [depth collision policy](#depth-collision-policy)).
  The robot may move into space it has not observed; the evaluator measures contact.
- Depth pixels in the robot's and held object's current guard voxels are removed before
  the planners and the guard's own map get them (spheres grown by √3 × `obstacle_padding`,
  2 cm voxels), so the arm leaves no occupied voxels where it was earlier. An
  object within about one voxel of the robot surface cannot be told apart from the robot.
  Removed pixels also do not clear old map cells, e.g. where the target stood before
  the pick; this makes the planner more careful, never less safe.
- The guard used 5 cm voxels until 2026-10-08. They rejected paths the planner kept 1 cm
  clear: a carried object passing 3.9 cm beside the clutter, the jaws 1-2 cm from a
  drawer front. A guard coarser than the planner's map only tests luck, so its cells
  are now 2 cm, like the planners'; any cell a surface touches still counts as occupied.
- Planning uses snapshots while physics continues. Base motion invalidates the
  result beyond declared tolerances. Simultaneous base/arm planning is not supported.
  The known-map diagnostic assumes declared geometry except its explicit obstacle
  relocation test; depth guards use new capture-time evidence.
- Trajectories are position samples with piecewise-linear interpolation and its
  velocity derivative. Admission checks episode, joints, start, limits and speed.
  This does not certify acceleration, jerk or continuous swept-volume clearance.
- The physical contact checks cover this test, not every shape or task. No grasp,
  payload attachment, contact-rich door control or collision-aware gripper closing
  is claimed. The gripper is fixed during planning.
- Runtime real-time factor depends on rendering, physics and other GPU processes.
  One short run is not a statistically controlled backend benchmark.

## Next qualification steps

Prefer MoveIt as the reference backend if these measured small-scene results
remain representative. Keep cuRobo as an explicit experimental selection. Before
using either in search episodes: validate the full collision model, add an
independent trajectory collision check, test blocked goals and narrow passages
across seeds, and reject stale maps/plans while the runtime continues stepping.
Then introduce known-pose pick/hold/release with measured object motion and
attached-object collision geometry. Grasp perception comes after execution is
reliable in that controlled setup.

## Historical comparison before full-geometry qualification (2026-10-04)

All six planner runs below passed; the grid and house no-planner baselines also
passed. Each row includes three plan requests followed by physical execution of
the last path, a Cartesian goal, cancellation and walking. These are not three
independent successful episodes.

| Backend | Scene / map | Planning wall time [s] | Walking RTF |
|---|---|---:|---:|
| MoveIt | Grid / known box | 0.49–0.57 | 0.320 |
| cuRobo V2 | Grid / known box | 5.95–7.37 | 0.311 |
| MoveIt | Grid / depth | 0.41–0.66 | 0.326 |
| cuRobo V2 | Grid / depth | 5.85–6.00 | 0.324 |
| MoveIt | House / depth | 0.32–1.19 | 0.213 |
| cuRobo V2 | House / depth | 6.46–9.49 | 0.173 |

No-planner walking RTF: grid 0.323, house 0.182. Other GPU activity and short
measurement windows prevent attributing differences to planner residency alone.
House depth-run tracking error stayed below 0.005 rad for both backends; neither
reported arm contact. Camera rate was 31.25 simulated Hz throughout. A sampled
direct joint path collided at 59/101 samples in the grid depth scene, confirming
that obstacle-aware planning was needed.

MoveIt was simpler to deploy in the existing ROS container and faster in these
experiments. cuRobo's small adapter does not remove its CUDA/version qualification
cost. No GPU planning advantage was demonstrated here; batching and tuned planner
configuration remain unmeasured. Both need the same substantial shared work on
collision models, timestamps, execution and validation.

The local `output_check/manipulation_planner_comparison.html` (gitignored, so not in
other clones) includes selectable
videos and links to raw JSON. Keep the original reports when comparing future
changes. Focused validation: 115 pytest checks passed; Ruff passed on changed code.

## Full-model qualification result

`output_check/manipulation_qualification_02/report.json` passed the known-map
check, including physical obstacle relocation and stopping. Its coverage audit
checked 94 collider/camera entries; alignment checked 20 links. Maximum arm
tracking error was 0.00333 rad and final Cartesian position error was 0.000857 m.
No unwanted arm contact was detected. Cameras remained at 31.25 simulated Hz.

`output_check/manipulation_coverage_preflight/report.json` passed the expected
unknown-space refusal at the current arm configuration, before requesting a plan.
The same coverage guard checks the full returned path if preflight succeeds.
Additional view acquisition remains an agent responsibility; automatic view
selection is not implemented.

`output_check/manipulation_full_curobo_01/report.json` failed to find a plan with
the larger conservative model. Cause (found 2026-10-07): the generated self-collision
ignore list skipped only directly joined links. The body and each upper leg have a
sphere-less hip link between them, and their spheres overlap by about 2.5 mm, so cuRobo
judged every pose invalid. Links the arm never moves (body, legs, cameras) are locked
during arm planning, so `prepare_model` now ignores all pairs among them.

cuRobo speed (2026-10-07): its collision check scans every slot of the obstacle cache,
used or not. The adapter had a fixed 4,096-box cache, so a 4-box plan took 18-30 s.
With the depth map in cuRobo's own voxel grid, the box cache is fixed at 32 slots and
each new planner warms up; a new gripper angle builds a new planner once (about 15 s).

Library depth maps, gate 2026-10-07 (`output_check/map3b_*`): pick, lift, refused and
allowed carry, place; 4/4 passed, 0 N furniture contact. Depth/center plans: MoveIt
0.04-0.21 s, cuRobo 0.12-0.13 s; refused carry: MoveIt uses its 30 s budget, cuRobo
3.4 s. Fusing the three images before a plan: MoveIt 0.46-0.50 s, cuRobo <= 0.01 s.
Before the clearance was added, cuRobo planned a carry with 0 mm between the held object
and the clutter top; the guard rejected it before motion (`output_check/map3_curobo_depth_center`).

cuRobo voxel bug (2026-10-07, commit above): a planner built with only boxes, after a
planner with the voxel grid, makes the voxel check of the earlier planner fail silently
inside its CUDA graphs; plans then pass through the depth map. Likely cause: Warp
rebuilds the shared collision kernels for the new obstacle type. The adapter gives every
planner the same obstacle types (boxes and voxel grid). Evidence: offline probes in this
session; with `use_cuda_graph=False` the same planner refused the same goals.

MoveIt notes: its Octomap self filter looks up the robot link frames in TF; the worker
publishes them with each image, so a lookup only waits for the message
(`shape_transform_cache_lookup_wait_time` 0.05 s). MoveIt (Jazzy) also segfaults while it destroys the
Octomap updater at interpreter exit; the worker sends every response first, then exits
with `os._exit(0)`. Planner timeouts and independently rejected paths remain preserved in
`output_check`; the checks do not retry with weaker geometry or switch backends.

Focused tests: 122 passed. The hidden-guide-collider regression was verified to
fail without its fix; source files were not changed during that negative check.

## Depth collision policy

Decision 2026-10-07: observed occupied voxels block motion; unknown voxels do not.
The robot may move into space it has not observed, as the real Spot and most
agent-facing systems do. Safety in unseen space is measured, not prevented: the
evaluator fails a run on any unwanted contact. Observing first (`look_at` before the
plan) still matters, because the planner can only avoid what the cameras see.

The earlier strict mode (`require_observed`: unknown blocks motion) was removed from
the robot test. It needed extra map rules for space the robot itself hides (swept
space, held-object history, contact regions), and those rules caused three of the
recent failures. The shared validator keeps its `require_observed` option.

```bash
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=47 ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_arm_planning.py \
  --spec scripts/PhyGS/tests/robot_tests/configs/episodes/table_reach_depth.yaml --planner moveit --record \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_depth_reach
```

Use a fresh output directory. Add `--viz kit` to watch.

Shared mapping retains per-camera timestamps. Earlier runs used two virtual
body-mounted cameras and a padded sphere self-filter. The virtual views put the
robot in 30–35% of each depth image; the onboard cameras show only the finger,
closer than 0.2 m (`output_check/real_camera_views_01`). Both were removed. Visibility and containment investigation probes have been removed.
No grasping, object attachment, or unknown-space safety guarantee is claimed.

### Verified observed-obstacle reach

`output_check/depth_reach_04/report.json` passed MoveIt joint reaching, public
Cartesian reaching, cancellation, and locomotion with depth-derived collision
boxes. Wrist position error was 1.15 mm; measured arm and obstacle contact forces
were zero; maximum tracking error was 0.00384 rad. Cameras delivered 31.25 simulated
Hz. Execution RTF was only 0.0127 and total wall time 424 s: this is behavior
validation, not a performance qualification or unknown-space safety certificate.

This run used the saved `depth_reach_04/experiment.yaml` with a 90.0 s planning
budget. The 30 s attempt timed out; default settings remain unchanged. The first
attempt was interrupted to replace the scalar occupied-voxel check with an
equivalent batched check. One subsequent setup attempt used an invalid integer
ROS planning-time parameter; its failed report is retained. cuRobo was not tested
in this change. Video: `output_check/depth_reach_04/third_person.mp4`.

To reproduce the tested configuration, add:
`--config scripts/PhyGS/output_check/depth_reach_04/experiment.yaml`
to the command above, with a fresh output directory.

### Historical depth-processing performance (virtual cameras)

The removed self-filter used SciPy's `cKDTree` to find pixels within robot spheres instead
of testing every pixel against every sphere. Sphere transforms are batched per
link. Static joint definitions are cached for up to eight URDF file versions,
keyed by resolved path, modification time, size, and inode. Each call still uses
current joint angles; file replacement invalidates the cache. Geometry, filter
padding, and collision policy are unchanged. Equivalence
tests compare both calculations with their previous scalar forms.

`depth_reach_indexed_01` passed the same physical checks: 0.62 mm wrist error,
zero measured arm/obstacle contact, and 31.25 simulated camera FPS. Execution RTF
was 0.0254 versus 0.0127 in `depth_reach_04`; total wall time was 266 s versus 424 s.
These runs have different planner paths/timing and shared hardware load, so this
is an observed comparison, not an isolated end-to-end benchmark. A separate
640×480 synthetic self-filter input produced identical masks in 0.12 s versus
0.81 s. Results and video are in `output_check/depth_reach_indexed_01`.
Performance remains inadequate for interactive wall-clock operation.

The subsequent static-kinematics cache run, `depth_reach_cached_01`, passed with
0.42 mm wrist error, zero measured contact, execution RTF 0.0466, and 156 s total
wall time. Its 100-call same-model benchmark reduced transform time from 5.74 ms
to 0.334 ms with matching results. File replacement and deletion are tested;
only static joint definitions are cached. End-to-end timings remain subject to
planner variation. The recording is `output_check/depth_reach_cached_01/third_person.mp4`.
