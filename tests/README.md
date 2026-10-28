# Tests

Two kinds of tests: pure Python unit tests (`tests/test_*.py`, no Isaac Sim) and robot tests
(`tests/robot_tests/`, a real simulator). Run everything from the Isaac Lab root
(`/workspace/isaaclab`). The project runs at 250 Hz physics.

## Unit tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest scripts/PhyGS/tests -q \
  --ignore=scripts/PhyGS/phygs_robot_tests -p no:cacheprovider
```

| Path | Purpose |
| --- | --- |
| `test_episode_runtime.py` | Runtime authority, atomic ownership, expiry and failure checks. |
| `test_locomotion_runtime.py`, `test_locomotion_launcher.py` | Policy write scope, runtime-backed locomotion commands, launch modes. |
| `test_manipulation_runtime.py` | Arm and gripper execution, ownership, measured completion, holding, contact chatter. |
| `test_manipulation_skills.py` | The `stow` and `look_at` skills and the skill lifecycle, with fake arms and planners. |
| `test_lowering.py`, `test_end_effector_execution.py` | End-effector trajectories: IK lowering, refusals, tool tracking. |
| `test_manipulation_planning.py` | Planner trajectory contracts and the shared 3D collision map. |
| `test_routines.py` | The robot tests' routines and the retrieve plan on a fake session. |
| `test_agent_imports.py` | Agents, routines and episode plans import only the session, the standard library, numpy and yaml. |
| `test_episode_spec.py`, `test_agent_session.py` | Episode specs and the agent session (grants, audit, threads). |
| `test_parts.py` | Sliding-part (drawer) information from an authored prismatic joint. |
| `test_navigation.py`, `test_navigation_backends.py` | Planning, command lifecycle, backends without ROS, independent evaluation. |
| `test_nav2_*.py` | Nav2 launcher, point-cloud geometry and ROS transport (needs ROS packages). |
| `test_footprint.py` | Transit footprint calibration and validation. |
| `test_observations.py`, `test_pipeline.py` | Observation permissions, capture-time transforms, processing pipelines. |
| `test_object_search.py`, `test_color_detection.py` | Object-search evidence and the reference color detector. |
| `test_scene_assets.py`, `test_scene.py` | Scene spawn-area validation and USD scenes. |
| `test_skill_api.py` | Public skill APIs import without manipulation planners or ROS. |
| `test_ui_*.py` | The robot UI (see below). |
| `fakes.py`, `ui_fakes.py` | Shared fakes (joint worlds, a scene for a real `IsaacLabWorld`, navigation, mapping, ROS transport; the UI's); test modules import only these, never each other. |

## Robot tests

Every robot test spawns a scene and has the robot do one thing; they form a ladder in the order the
framework is built. The ladder, the two test shapes (agent tests and component tests), how to run
it and how to add a rung: [tests/robot_tests/README.md](../tests/robot_tests/README.md).

```
robot_tests/
  README.md        the ladder: every robot test in build order
  common.py        shared start and result of every robot test; the robot as episodes build it
  physical.py      the robot's ground-truth state for the navigation evaluators
  run_suite.py     runs the ladder or a group of it, with a summary and the first failing rung
  routines/        session-only building blocks of the agent tests' plans (params.yaml: their strategy)
  episodes/        agent tests: an episode spec and a plan through the agent session (episode.py: their test side)
  components/      component tests: the machinery under the agent session (RobotControl or the runtime)
  configs/         test settings and episode specs; the robot is --robot (default: the spec's)
```

`--robot` (a robot name or a `robot.yaml`) runs a robot test with another robot ([adding a robot](../docs/adding_a_robot.md)).
The unit tests' description check (`tests/test_robot_description.py`) is the ladder's first rung:
every robot of `phygs/robots/` against its URDF and USD, no simulator.

Every robot test takes `--output DIR` (a fresh folder; use `output_check/<YYYYMMDD-HHMM>_<name>`),
runs automatic and headless by default (`--viz kit` to watch, `--manual` for keyboard control
where a test has it), writes `DIR/check.json` (`test`, `status`, `error`, `checks` and the test's
data), prints `<test>: <status>; <DIR>` and exits 0 only when it passed. A hang leaves every
thread's stack in `DIR/stacks.txt` (`kill -USR1 <pid>`). `--record` writes `DIR/third_person.mp4`
(agent tests: the scene's record view; component tests: a camera following the robot; the world
lifecycle test has no video). Its overlay shows the test, the simulated time, the real-time factor
(RTF) and the frame rates (simulated and per wall second), the latest skill step of each kind, and a
timeline of every command on a BASE and an ARM lane (blue running, green succeeded, red failed,
grey cancelled) with a playhead. Beside the scene view, a grid of panels (`helpers/panels.py`;
"no data" when the run has none):

| Panel | Shows |
| --- | --- |
| FRONT, FRONT DEPTH | The two front cameras stitched into one forward view (as in the UI: `helpers/front_view.py`; a virtual camera between them, tilted down like them, 130 x 100 deg, keeping 90% of their pixels and the ground ahead of the feet) and its distance (blue near, red at 6 m and beyond). A white line marks where the cameras' fields of view end. |
| HAND, HAND DEPTH | The hand camera and its depth. |
| NAVIGATION | Nav2's global costmap in a 6 m window around the robot, map +x right and +y up, coloured as RViz draws Nav2's costmap in its "costmap" scheme: free transparent (RViz's grey background), cost blue to red, inscribed cyan, lethal magenta, unknown grey-teal. Nav2's route (Smac's path, green), the goal and its heading (white), the footprint rectangle and heading (yellow), the commanded velocity (cyan, 1 s of travel). |
| ARM PLANNER | The last arm plan in the body frame, seen from the robot's right and framed on the arm's motion: the arm at the start (grey), along the plan (faint) and at the end (orange, with its collision spheres), the jaw's path (magenta), cuRobo's obstacle voxels from the depth cameras coloured by distance to the arm (red within 5 cm, blue from 30 cm), known boxes (outlined) and the floor (a grid). Its label: plan time, result, planning time, voxel count. |

Agent tests capture the front and hand cameras for the panels whether or not the agent observes
them. Component tests show the cameras the control observes, and the planner when the test passes it
(`check.show_planner`); a test with none of these at its first frame (for example
`check_manipulation`, `check_observations`) records the scene view alone.

```bash
# The ladder up to a rung (ROS 2 Jazzy and explore_lite sourced for rungs 6, 7, 11-15). --record adds
# the videos; open <folder>/index.html for the results and every video.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/run_suite.py ladder --upto 10 --record

# One test.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_retrieve.py \
  --spec scripts/PhyGS/tests/robot_tests/configs/episodes/dev_house_drawer.yaml \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_retrieve_house
```

An agent test's folder also holds the harness's `report.json` (spec, how the episode ended, the
evaluator's result (`evaluation`), call totals and privileged reads, commands with their final feedback, arm
contacts, narration) and `audit.jsonl` (one line per agent call). Thresholds:
`configs/<test>.yaml` and `configs/episodes.yaml` (in `phygs_robot_tests`); the plans' strategy:
`routines/params.yaml`.

Notes kept from validation:

- The world lifecycle check runs real PhysX with fixed leg posture gains and no walking policy.
  Each cycle first tests an immediate reset, then a settled episode, so evaluated episode IDs are
  2, 4 and 6. The simulated emergency stop stops the episode; it is not a hardware e-stop.
- On depth (`table_reach_depth`) the arm planner's observed occupied voxels block motion and
  unknown voxels do not; the test side counts any arm contact outside the target from the
  episode's `report.json` (`arm_contacts`).
- The house search case is bounded to the kitchen (`exploration.bounds`); unrestricted
  whole-house frontier search is not a validated configuration.

## Episodes (examples/run_world.py)

One entry point runs an episode spec with an agent (design:
[docs/design/episode_harness/index.html](../docs/design/episode_harness/index.html)). Headless by default;
`--viz kit` opens a window, `--record` writes `third_person.mp4`, `--clock realtime` keeps time
running while the agent thinks.

```bash
./isaaclab.sh -p scripts/PhyGS/examples/run_world.py \
  --spec scripts/PhyGS/tests/robot_tests/configs/episodes/dev_grid_table.yaml --agent template \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_grid_table
```

## Robot UI

A local browser UI for a person to run Spot (design: [docs/design/ui/design.html](../docs/design/ui/design.html)).
The simulation runs headless; the page shows only the agent view: the Front view stitched from
the two front cameras, the frontleft, frontright and hand cameras (RGB or depth), the map built
from the front cameras' depth, the robot state, skill step lines, commands and a timeline. Its
E-stop ends the episode. Tiles run Navigate to (click a goal on the map, drag for a heading),
Drive (hold W A S D Q E or the pad) and Stop; a running command can be cancelled. Arm joints,
Joint path, Stow, Open and Close move the arm and gripper; Look at and Move to pose take a point
clicked in a camera view; Move along runs a straight wrist line in a direction. With
`arm.backend: planned` (default) the arm plans collision-free paths on the robot's own depth
cameras (the core `ArmPlanner`, `execution/manipulation/arm_planner.py`). Arm joints and Move to pose can Preview their plan. Drive has
Look around (one turn on the spot; unknown floor is blocked, so start with it). The Telemetry
button (key T) opens a drawer over the status column: health (speed vs real time, runner and map
timing, planner, cameras, link), base and arm charts, joint limit margins, the mission, and any
command's raw inputs and feedback (also from a click on the log or timeline). The output folder
gets `telemetry.jsonl` (5 Hz of sim time).

```bash
./isaaclab.sh -p scripts/PhyGS/phygs/ui/cli.py \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_ui_grid
# Open the printed http://127.0.0.1:8765/?token=... in a browser on this machine.
# Remote machine: ssh -L 8765:127.0.0.1:8765 <host>, then open the same URL locally.

# Replay an episode in the browser (no Isaac Sim):
./isaaclab.sh -p scripts/PhyGS/phygs/ui/replay_cli.py \
  scripts/PhyGS/output_check/<YYYYMMDD-HHMM>_ui_grid
```

`--spec <file>` runs another spec, e.g. `phygs/configs/ui/house_drawer.yaml`; `--demo tour` drives a
short fixed route; `--autostart` starts without the page. Unit tests: `test_ui_stitch.py`,
`test_ui_translate.py`, `test_ui_session.py`, `test_ui_episode.py`,
`test_ui_replay.py`, `test_ui_previews.py`, `test_ui_telemetry.py`.

## Standalone scene loading

```bash
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/load_usd_asset_standalone.py \
  --usd_path scripts/.scene_usd/bear_loop_house_revised/scene.usdc \
  --device cpu --viz kit --physics_dt 0.004166666666666667 --no_render_every_step
```

Optional `--validate_scene` checks use `scene_generation/usd_tools/scene_lib/scene_checks.py`. For a bounded
headless check, use `--headless --viz none --render_mode none --num_steps 1200 --validate_scene`
with the same physics timestep. House-generation and other scene checks live under
the scene tools' own tests (`scene_generation/usd_tools`).

## ForageBench scene setup

Authenticate once with access to the private dataset:

```bash
./isaaclab.sh -p -m huggingface_hub.cli.hf auth login
```

Discover available scenes, then download one complete scene:

```bash
./isaaclab.sh -p scripts/PhyGS/phygs/scenes/fetch.py list

./isaaclab.sh -p scripts/PhyGS/phygs/scenes/fetch.py fetch \
  --scene_id bear_loop_house --scene_version 1.0.0 \
  --cache_dir scripts/.scene_usd/cache
```

Fetch downloads the manifest, scene and textures, verifies the archive, and creates an extracted
cache and scene record. It prints the USD path, resolved repository commit, and **Scene record**
path. Use that printed record path to load offline:

```bash
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/load_usd_asset_standalone.py \
  --scene_record /PATH/PRINTED/BY/FETCH.json \
  --device cpu --viz kit --no_render_every_step --report /tmp/bear_loop_run.json
```

Records live under `CACHE/records/COMMIT/SCENE-VERSION.json`. The record supplies its USD path and
default physics timestep. The report includes dataset ID, commit, scene version, checksum, and
actual timestep. An explicit `--physics_dt` overrides the scene default and is recorded separately.

Download every catalogued scene release with `fetch --all --cache_dir scripts/.scene_usd/cache`;
for a subset, pass several IDs after `--scene_id`. `--repo_id` defaults to
`mandmlabresearch/ForageBench`; pass `--revision FULL_COMMIT_SHA` to reproduce a previous
download. Do not modify or commit cached files or generated records.

For benchmark configurations in this repository, the checked-in YAML pins a version:

```bash
./isaaclab.sh -p scripts/PhyGS/phygs/scenes/fetch.py fetch \
  --config scripts/PhyGS/configs/scenes/bear_loop_house.yaml \
  --cache_dir scripts/.scene_usd/cache \
  --record scripts/.scene_usd/cache/bear_loop_house.json
```

Scene release preparation and upload remain in `scripts/scene_generation/usd_tools`; see
`scripts/.scene_usd/foragebench/PIPELINE.md`.

## Test philosophy

- Unit tests isolate API contracts and deterministic helper logic.
- Robot tests validate real integration and fail loudly; no silent fallbacks.
- If a test cannot run in a lightweight environment, document the missing runtime instead of faking behavior.
