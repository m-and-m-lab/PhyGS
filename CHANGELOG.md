# Changelog

All notable changes to Interactive Search. Newest first.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project has no
version tags yet, so entries are grouped by date.

How to add an entry: put one line under **Unreleased** in the right group
(Added, Changed, Deprecated, Removed, Fixed). Use past tense and name the public
symbol or file. For Changed, Deprecated and Removed, say what to use instead.
When you commit a batch, move its lines under a new dated heading.

## [Unreleased]

### 0.6.0 (unreleased, develop)

The task evaluator samples the run (ForageBench's phase C, 2026-10-29). A change to the `Task`
protocol, so a minor release; the report's keys are unchanged (schema 2).

Changed:

- **Breaking:** `Task` has `reset(truth)` (the settled world, before the agent starts; called only
  when the start check passed), `update(truth)` (every `evaluation_period_seconds` of simulated time
  while the agent runs) and `result(truth, end)`. `end` is the new `phygs.world.EpisodeEnd`
  (`ended_by`, `sim_seconds`, `contacts`), in place of `result(truth, ended_by, contacts,
  drawers_open_at_start)`; a task that needs the drawers' openings at the start reads them in
  `reset`. Migrate a task by adding the two methods (no-ops are fine) and reading `end.ended_by` and
  `end.contacts`.
- The robot tests' `RetrieveTask` and `NoTask` follow the protocol; their results are unchanged.

Added:

- `Truth.arm_contacts(threshold_n)`, `Truth.pinched(name, threshold_n)` (every jaw link presses on
  the body), `Truth.started_inside(name)` (the drawer a body started in).
- Setting `evaluation_period_seconds` (0.2 s).

### 0.5.0 (2026-10-29)

Run statuses and the check before the agent starts (ForageBench's phase A, 2026-10-29). Report
schema 2, so a minor release (docs/world.md, Versions). The repository holds the framework only
(phase B, 2026-10-29): scene-building code and design history left it.

Removed:

- **Breaking:** `scene_generation/` (the Infinigen-Indoors fork, the USD scene builders, the
  packager and publisher, the generation notes). PhyGS loads scenes as released USD files by id and
  version and does not build them; the scene tools are kept with the lab's scenes, and the generator
  of the paper is on the `main` branch. `docs/scenes.md` keeps the release format PhyGS reads.
- `tests/robot_tests/load_usd_asset_standalone.py`, a scene-inspection tool that went with them.
- `docs/design/` (the design pages and their history). The docs describe the code as it is; the robot
  UI's page became the guide `docs/ui.md`, and links to the pages point at the guides.

Added:

- `docs/ui.md`: the robot UI guide (running it, what it shows and never shows, running a skill, what a
  run leaves, replay, the protocol).
- `tests/test_packaging.py`: `pyproject.toml` lists every package in the tree (the list stays explicit
  because the robot tests install as `phygs_robot_tests`).
- `.vscode/` is ignored; untrack the committed `settings.json` with `git rm --cached`.

Added:

- The check before the agent starts (`phygs/world/start.py`, settings `start_check`): after the
  reset's settle, the robot must stand (base within 15 degrees of vertical, above its fallen height),
  every scene body must be at rest (under 0.1 m/s) within 0.05 m of where the episode placed it, and
  every drawer closed (under 0.03 m; `drawer_open_max_m: null` lets drawers start open). When it
  fails, the World writes the report with `ended_by: setup_error` and never starts the agent.
  Measured on every robot-test scene; the house reaches 0.026 m/s and 0.023 m, the others 0.
- `Truth.body_motion()`: each scene body's speed and distance from where it was placed.

Changed:

- Report schema 2: `status` (`completed`, `setup_error` or `error`, from `phygs.world.report.status`)
  says whether an episode can be scored; only `completed` episodes belong in a rate. `error` is the
  simulator or the World breaking (`environment_error`, `runtime_failed`); an agent's crash, a fall
  or the budget is `completed`, the agent's failure. `start_check` records the check's result,
  measured values and limits.
- An arm planner process that breaks (out of GPU memory, a crash, a timeout; not a plan it could not
  find) fails the runtime, so the episode ends `runtime_failed` with status `error` instead of being
  scored as the agent's failure (`ArmPlanner.failure`). Seen 2026-10-29: a CUDA out-of-memory in the
  planner while another simulator held the GPU was reported as `agent_failed`.

### 0.4.0 (2026-10-29)

The layout of the layout proposal, steps 1 to 4 (`docs/design/framework_layout/`; the differences
are on its first page). No behaviour, setting, spec key or report key changed. **Breaking:** the old
module paths are removed in this release, without a deprecation period (decided 2026-10-29); the
table under Removed gives each new path.

Changed:

- The packages are layers that import downward only: ui, world, session, robots, skills, execution,
  sim and recording, scene and runtime, observations (`docs/layout.md`). `tests/test_layers.py`
  enforces it. The runtime import cycles through `simulation/` are gone.
- **Breaking:** `Environment` is `World` (`from phygs.world import World`); `phygs.world` exports
  `World`, `EpisodeSpec`, `Truth`, `Task`, `NoTask`, `load_task`, `load_agent`, `run_episode`,
  `SETTINGS` and the schemas, each imported on first use.
- The World's module is split by what it does: `world/world.py` (the class), `world/build.py` (the
  steps that need no simulator: settings and overrides, refused combinations, cameras, observation
  access; unit-tested now), `world/truth.py` (`Truth`), `world/report.py` (`compose` refuses a
  report whose keys differ from `KEYS`).
- Agents import `from phygs.session import ...` (the package re-exports the session's types).
- The example agents are `examples/agents/` (`template`, `depth_example`); `--agent template` and
  `load_agent("template")` find them there.
- `ArticulatedPart` is defined in `phygs.scene.types`; `phygs.session` and the skills' types re-export it.
- The robot tests set the planner backend through `World(overrides=...)` instead of a settings copy.
- Docker: `docker/Dockerfile.phygs` with `docker-compose.phygs.patch.yaml` builds Isaac Lab's `ros2`
  service as what PhyGS runs on: Isaac Lab 3.0.0's base image (Isaac Sim 6.0.1), ROS 2 Jazzy with Nav2,
  explore_lite, MoveIt, cuRobo V2 and the editable PhyGS install (`docker/README.md`). The image is
  written to match this machine's container; it has not been built from scratch yet.
- `planner.curobo_dependencies` is `/opt/curobo_deps` (was `/tmp/spot-planner-deps`, which a new
  container loses).

Removed (**Breaking**), old → new:

| Old | New |
| --- | --- |
| `phygs.episodes.session`, `phygs.episodes.hints` | `phygs.session.session`, `phygs.session.hints` |
| `phygs.episodes.environment` (`Environment`) | `phygs.world.world` (`World`) |
| `phygs.episodes.spec`, `.task`, `.viewer`, `.recorder`, `.provenance`, `.report` | `phygs.world.spec`, `.task`, `.viewer`, `.recorder`, `.provenance`, `.report` |
| `phygs.episodes.scenes` | `phygs.scene.registry` |
| `phygs.scenes.fetch` | `phygs.scene.fetch` |
| `phygs.simulation.cameras`, `.fabric`, `.observations`, `.base`, `.arm`, `.footprint` | `phygs.sim.cameras`, `.fabric`, `.observations`, `.base`, `.arm`, `.footprint` |
| `phygs.simulation.world` | `phygs.sim.state` |
| `phygs.simulation.scene`, `.parts`, `.scene_assets`, `.storage_units` | `phygs.scene.reader`, `.parts`, `.assets`, `.storage_units` |
| `phygs.simulation.maps` | `phygs.execution.navigation.known_map` |
| `phygs.simulation.navigation` | `phygs.world.navigation` |
| `phygs.tools.front_view`, `.color_detection` | `phygs.observations.front_view`, `.color_detection` |
| `phygs.tools.video`, `.panels`, `.timeline`, `.narration` | `phygs.recording.video`, `.panels`, `.timeline`, `.narration` |
| `phygs.tools.collision_map` | `phygs.execution.manipulation.collision_map` |
| `phygs.agents.template`, `.depth_example` | `examples/agents/template.py`, `depth_example.py` |

Also removed:

- `tools/sync_from_interactive_search.py`: PhyGS is developed in this repository now; a change made
  in the lab's interactive-search tree is ported by hand.
- `docker/Dockerfile.curobofix` and its compose patch (the May beta's container: Isaac Sim 5.1, the
  cuRobo fork, AO-Grasp tools), replaced by `docker/Dockerfile.phygs` (see Changed).

Fixed:

- `pyproject.toml` lists `phygs.execution.manipulation.skills`, which a non-editable install missed.
- `docs/world.md` no longer names a session schema number; the session has none.

### 0.3.2 (2026-10-29)

- Synced from interactive-search (2026-10-29, `tools/sync_from_interactive_search.py`): Ridgeback-Franka's
  `robot.yaml`, actuators and README; Ridgeback support in the cuRobo, MoveIt and model planners; the
  table-reach episodes' per-robot starts; the pick-place config; `tests/test_second_robot.py`; the
  adding-a-robot guide and design page.
- The `phygs` command (`python -m phygs`; console script with the editable install): `run`, `ui`,
  `ui replay`, `scenes`, `tests`, `version`, each dispatching to the existing script.
- Report: `episodes/report.py` with `REPORT_SCHEMA = 1` and the key list; every report records
  `schemas: {spec, report}`.
- `Environment(overrides=...)`: a mapping applied over `configs/environment.yaml` (one-level merge), for
  benchmarks that pin end-of-episode and backend settings. `environment.yaml` names those keys.
- `docs/world.md` (the benchmark author's page), `docs/layout.md` (the layers and the import rule) and
  the layout proposal `docs/design/framework_layout/` (step 0 of it is this entry).

### Repository split (2026-10-28)

PhyGS `develop` now holds the simulation framework copied out of the lab's internal
interactive-search tree (its working state of 2026-10-28, after the retrieve evaluator and the task
seam). The May beta (`phygs_simulation/`, AO-Grasp, the old skills) stays on `main` until this is
released. Every rename, old → new:

| Old (interactive-search) | New (PhyGS) |
| --- | --- |
| `src/interactive_search/` (package `interactive_search`) | `phygs/` (package `phygs`) |
| `scripts/skills/{locomotion,navigation,manipulation}/api.py` (package `skills`) | `phygs/skills/...` (`phygs.skills.*`) |
| `src/interactive_search/helpers/observations/` | `phygs/observations/` |
| `src/interactive_search/helpers/{video,panels,timeline,narration,color_detection,collision_map,front_view}.py` | `phygs/tools/` |
| `scripts/helpers/foragebench.py` | `phygs/scenes/fetch.py` (`./isaaclab.sh -p -m phygs.scenes.fetch`) |
| `scripts/navigation/nav2.launch.py` | `phygs/execution/navigation/launch/nav2.launch.py` |
| `configs/` | `phygs/configs/` |
| `spot_model/` | `phygs/robots/spot/assets/` (`spot_description` stays a submodule) |
| `policies/` (unused checkpoints) | dropped; the deployed policy is `phygs/robots/spot/locomotion/` |
| `scripts/run_episode.py` | `examples/run_world.py` (`--task_loader module:function`) |
| `scripts/ui_episode.py`, `scripts/ui_replay.py` | `phygs/ui/cli.py`, `phygs/ui/replay_cli.py` |
| `src/interactive_search/tasks/`, `scripts/tasks/object_search/api.py` | `tests/robot_tests/tasks/` (the robot tests' own evaluators; the framework imports no task) |
| `tests/robot_tests/` | `tests/robot_tests/` (an importable package; `routines`, `episode`, `common` are `phygs_robot_tests.*`) |
| `../.scene_usd/{grid_storage,nav_course,dynamic_course}` | `tests/robot_tests/scenes/` (fixtures) |
| `../.scene_usd/tools/` | `scene_generation/usd_tools/` (without `legacy/` and the 61 MB `assets/`) |
| `docs/{episode_harness,manipulation_contract,robot_portability,evaluator_and_robot_tests,framework_benchmark_split,ui,visual,archive}` | `docs/design/...` |
| `docs/0*_*.md`, `legacy_scene_generation_notes.md` | `scene_generation/docs/` |
| `third_party/ao-grasp`, `third_party/curobo` submodules | dropped (nothing imports AO-Grasp; cuRobo is cloned by `docker/Dockerfile.curobofix`) |

Not copied: `output_check/`, `reports/`, `research_notes/`, the benchmark design pages
(`docs/benchmark_*`, `docs/repo_split`), the presentation sites and the asset catalogs' images.
Scenes other than the fixtures are found through `$PHYGS_SCENES` (default `~/.cache/phygs/scenes`).

Scenes and versions (2026-10-28): a spec may pin a fetched release, `scene: {id: bear_loop_house,
version: 1.1.0}`; the registry checks the version against the fetch record. The house specs pin
`bear_loop_house` 1.1.0 (the storage revision, packaged and prepared for the dataset the same day);
`scene_generation/usd_tools/publish_scene.py` replaces the one-release publisher and adds releases to
an existing catalog.

Later changes made in interactive-search came here by a sync script (copy and rename; never the pinned
policy files) until 0.4.0, which removed it: PhyGS is developed here now.

The entries below are interactive-search's, kept as the framework's history.


### Changed

- **Breaking:** episode spec schema 3 (2026-10-28). Scenes are named by id (`scene: {id: grid_storage,
  ground: true}`, optional `name`); `episodes/scenes.py` finds the file under `$PHYGS_SCENES`,
  `~/.cache/phygs/scenes`, then the lab's `.scene_usd` folder. `scene.usd` and any `..` path are
  refused. `tier` is replaced by `actions`, which takes a named set (`skills`, `motion`, `joints`) or
  a list. The task block is opaque to the framework: `EpisodeSpec.task` is the mapping as written,
  `TaskSpec` and `GOALS` left `episodes/spec.py`; `phygs_robot_tests.tasks.load_task(spec.task)`
  builds the task object the `Environment` now takes (`task=`). The report's `spec` loses `goal`,
  `description` and `tier`; a `task` key (goal, description) is added beside it.
- `tasks/retrieve.py`: `RetrieveEvaluator` became `RetrieveTask(description, target)`, which also
  gives the brief the robot UI shows and the provenance entry. `Environment.evaluator` is
  `Environment.task`.

### Added

- The task seam (2026-10-28): `episodes/task.py` (`Task` protocol, `NoTask`); a guard test
  (`tests/test_framework_imports.py`) that no framework module imports the tasks package and no spec
  holds a `..` path.
- Retrieve evaluator (2026-10-28): `tasks/retrieve.py` `RetrieveEvaluator` replaces the placeholder
  judge. Called once after the robot has stopped, with ground truth: the outcome (`success` when the
  target is held, the arm stowed, the robot upright and the episode ended in time) and, apart from it,
  the conduct (unintended arm contact at or above `contact_n`, drawers the agent left open) with
  `clean`. Thresholds in `tasks/retrieve.yaml`, hashed into the report's provenance
  (`evaluator_thresholds`). The report key `judge` is now `evaluation`; `check_retrieve.py` reads it.
  `Truth.held` takes the evaluator's thresholds (`lift_m`, `reach_m`); `Truth.part_openings()` added.
- Stop at the end of an episode (2026-10-28): after the agent ends, the Environment cancels the
  navigation goal and the arm command (or stops direct walking), keeps the grip, ticks until every
  controller is idle or `stop_all_timeout_seconds` (3 s), then settles `end_settle_seconds` and
  evaluates. The report's `stopped` says what was cancelled and how long it took. New
  `NavigationClient.active_command`, `ManipulationClient.active_command`, `CommandClient.name`.
- Robot-test ladder (2026-10-28): every robot test in build order, 15 rungs from scene and reset to
  the public entry point (`tests/robot_tests/README.md`; `run_suite.py ladder --upto N`, `--from N`).
  The summary names the first failing rung, the lowest broken module. Agents' requests have agent
  tests (a spec and a plan through the session), the machinery under them component tests.
- Agent tests (2026-10-28): `episodes/check_navigation.py` (specs `nav_course`, `nav_house`),
  `episodes/check_dynamic_obstacle.py` (`nav_dynamic`), `episodes/check_arm_planning.py` and
  `episodes/check_pick_place.py` (`table_reach`, `table_reach_depth`; `--planner curobo|moveit`),
  `episodes/check_entry_point.py` (the template agent through `examples/run_world.py`), and the
  course scenes `.scene_usd/nav_course` and `.scene_usd/dynamic_course`
  (`scene_generation/usd_tools/build_test_course_scenes.py`).
- `Environment.run(agent, during=...)`: a callable the harness calls every tick with the
  Environment, so a robot test can sample ground truth or move a scene body during an episode
  (2026-10-28).
- The template agent makes one call on each client its spec's tier grants (a body velocity and an
  arm joint move on the joints tier; spec `joints_grid`) (2026-10-28).
- Robot descriptions (2026-10-28): every robot fact in one `robot.yaml` per robot folder
  (`phygs/robots/<name>/`), loaded and checked by `RobotDescription`
  (`robots/description.py`: `load_robot`, `validate_urdf`, `validate_usd`); camera roles
  (`navigation`, `planning`, `look`, `front`); `session.robot` (`RobotInfo`) for agents; `--robot` on
  `run_world.py`, `cli.py` and the robot tests. Guide: `docs/adding_a_robot.md`; template:
  `robots/_template/robot.yaml`.
- Base controllers (2026-10-28): the `BaseController` contract (`execution/locomotion/base.py`, with
  `DeadBand` and `Deceleration`), `PlanarJointBase` (`planar_joints`: velocity-driven x, y, yaw joints)
  beside the walking policy (`legged_policy`), or a plugin `module:Class`.
- `tests/robot_tests/components/check_bring_up.py --robot <name>`: a new robot's first simulator check
  (spawn stowed, drive, arm, gripper, cameras, localization), every value from its `robot.yaml`;
  the README and `docs/adding_a_robot.md` give the bring-up steps (2026-10-28).
- `robots/ridgeback_franka/`: the second robot, in bring-up (description check, a simulator smoke of
  the base, arm, gripper and cameras, its measured transit footprint and the navigation course pass).
  `fetch.py` downloads its USD with pinned hashes (2026-10-28).
- Two-finger grippers: `ManipulationController(gripper_followers=...)` drives joints that follow the
  primary gripper joint; `MotionConfig.gripper_tolerance` and `gripper_stopped_velocity` in the
  gripper's units (2026-10-28).
- Prismatic joints in the arm chain (`ArmKinematics`, `link_transforms`) (2026-10-28).
- `tests/test_robot_neutral.py`: fails when Spot's joint, link, camera or class names appear in
  framework code outside `robots/spot/` (2026-10-28).
- Added `docs/design/robot_portability/index.html`: the design for adding other mobile manipulators (2026-10-28).
- Added `docs/benchmark_positioning/audited_review.html` with a primary-source comparison and implementation priorities for PhyGS and ForageBench (2026-10-27).
- `phygs/configs/environment.yaml` cameras `left`, `right` and `rear` (Spot's side and rear Gemini 2
  body cameras). Nav2 does not use them yet (`point_cloud.cameras`) (2026-10-27).
- `check_nav2_pipeline.py --camera_view`: `camera_view.mp4`, the first camera's images beside the
  costmap with every camera's field of view (2026-10-27).
- `manipulation.unstow()`: unfolds the arm from the stow to the pre-stow posture (an Unstow tile in
  the UI). `move_to_pose` and `plan` work from the stow: the arm planner starts the plan at the
  pre-stow posture and the trajectory first moves straight there (`ArmPlanner(stow=...)`)
  (2026-10-26).
- `ArmPlanner.warm_up`: the episode plans once with the measured and the open gripper angle before
  its clock runs; cuRobo's first plan per gripper angle took about 8 s (2026-10-26).
- `navigation.explore(timeout=None)`: frontier exploration as one navigation command. explore_lite
  (m-explore-ros2, built from source: README) picks frontiers on Nav2's global costmap; the bridge
  serves its `NavigateToPose` goals, each a leg followed like `navigate_to`. Needs a map prior with
  unexplored space: `DeclaredFloorMap(explored=(x, y, radius))`. Settings: `Nav2Config.explore_*`
  (2026-10-26).
- `Nav2Config.sensor_voxel_size` (`navigation.point_cloud.voxel_size`, 0.05 m): the bridge sends
  one point per voxel, their centroid (2026-10-26).
- Recorded videos (`--record`) show panels beside the scene view (`helpers/panels.py`): the
  stitched front cameras and the hand camera with their depth, Nav2's costmap with the route,
  goal, footprint and command, and the last arm plan (cuRobo's obstacle voxels, collision spheres
  and the jaw's path). See `tests/README.md` (2026-10-25).
- The stitched front view outlines where the cameras' fields of view end, in the UI and the video
  (`FrontView.outlined`) (2026-10-25).
- `PlanRequest.world_path`: cuRobo saves the obstacle voxels each plan avoids
  (`planner/world/*.npy`); `ArmPlanner.last_view` and `Nav2NavigationBackend.route()` for
  display (2026-10-25).
- Episodes without the `known_map` grant plan the arm on the robot's depth cameras
  (`planner.cameras` in `configs/episodes/environment.yaml`); manipulation now needs only
  `gt_localization` (2026-10-24).
- `check_manipulation_planning.py` checks the held-object carry refusal on known boxes only: on
  depth the clutter's top is unseen while the robot holds the box (2026-10-25).
- The MoveIt planner worker reports ready only once its Octomap listens to the depth points; an
  early plan's fusion timed out (2026-10-25).
- The MoveIt planner worker runs on a ROS domain of its own (215-232, one per worker): its robot
  link frames in `/tf` no longer reach Nav2 or another planner (2026-10-24).
- `execution/manipulation/arm_planner.py` (`ArmPlanner`): one arm planner host for the episodes,
  the robot UI and `check_manipulation_planning.py`, with known boxes (`boxes=`), the robot's depth
  cameras (`cameras=`) or both; `status()` for telemetry. `PlanRequest.images` carries the depth
  for one plan; the planner rebuilds its map from them and removes the robot's and the held
  object's pixels with the library's self filter (cuRobo `RobotSegmenter`; MoveIt's Octomap shape
  mask, with the robot's link frames now in TF). New setting `self_filter_padding` [m], 0.05 (2026-10-23; 0.02 left gripper edge pixels in the map, 2026-10-24).

- Robot UI: a Telemetry drawer (button or key T, closed by default) with health, base, arm,
  mission and command tabs from the agent view; `telemetry.jsonl` in the output folder
  (`UiSession(health=, telemetry_path=, telemetry_period=)`, `OnboardPlanner.status()`); Look
  around in Drive (one turn on the spot); the launcher prints how to open the link from another
  computer and the page explains a wrong token, a second browser or an unreachable server
  (2026-10-23).
- `MotionFeedback` of a finished arm or gripper command (succeeded, failed or cancelled) reports
  where it ended: `reached_seconds` of its reference, `joint_targets` (the reference then),
  `joint_positions` (measured), `stalled` (the arm stopped following its reference: a tracking
  failure or timeout), `gripper_aperture` (the measured gripper joint) and `tool_pose` (the measured
  hand pose; None without kinematics or localization). A finished skill (`stow`, `look_at`) reports
  the same end state in its `SkillFeedback` (2026-10-23).

- The episode report's `provenance`: the spec schema, SHA-256 of the robot and scene USD over
  every layer they compose (a `spot_physics.usd` edit changes the robot's), of the policy and its
  configuration, and the git commit with a `dirty` flag (`episodes/provenance.py`) (2026-10-23).

- Robot test videos: every robot test takes `--record` (`DIR/third_person.mp4`; component tests use
  a camera following the robot, `simulation.cameras.ViewCamera`), and `run_suite.py --record` adds
  `index.html` with every result and video. The overlay (`helpers/video.py`) shows the test, the
  simulated time, the real-time factor, the frame rates, the latest skill steps and a timeline of
  every command on a base and an arm lane by status (`helpers/timeline.py`; `SpotControl.timeline`
  records each client command call) (2026-10-18).
- `ArticulatedPart.floor_center`: the middle of the drawer floor the opening exposes now (house-unit
  drawers), in `session.hints.part(name)`; `check_drawer_place.py` places there (2026-10-18).

- `execute_trajectory` runs end-effector trajectories (`EndEffectorTrajectory`: timed tool poses,
  world or body frame, tool `hand` between the jaws or `wrist`) besides joint trajectories
  (`execution/manipulation/lowering.py`): solved with IK before moving and refused
  (`TrajectoryRefused`) when out of reach, needing another arm configuration or not starting at
  the tool; slowed down uniformly to the joint speed limit (joint trajectories too, instead of
  refused); the tool is tracked while it runs (2 cm, 0.05 rad, `MotionConfig.tool_*_tolerance`).
  Feedback adds `progress` and `time_scale`. Needs `ManipulationClient.bind_kinematics`; no
  collision check (design: `docs/design/manipulation_contract/index.html`, 2026-10-17).
- `navigate_to` finishes near its goal with single-axis moves
  (`execution/navigation/fine_approach.py`, used by the built-in and the Nav2 backend; not a skill):
  within `NavigationConfig.fine_approach_radius` (0.10 m) of a goal with a smaller position
  tolerance, the robot stops and, from each settled pose, turns to the goal heading or walks
  forward/back or sideways at no less than the lowest walking speed, until the goal is reached or
  the time budget ends; the fine approach keeps the goal until a settled pose is beyond
  `fine_approach_exit_radius` (0.25 m). The walking policy stood still under small commands that
  mix axes, so tolerances below 0.05 m timed out; with fine moves 0.03-0.05 m goals succeeded 5/5
  at 500 and 250 Hz (open floor, `output_check/20261016-2029_fine_approach_hysteresis`) (2026-10-16).
- Scenes load from their USD file and a config: `phygs.simulation.scene.UsdScene`
  finds objects (rigid bodies), drawers (prismatic joint with a `handle`), colliders and where an
  object starts (inside a drawer, or on a support) in the file by name; the arm planner's
  obstacles are the colliders around the robot. The grid storage scene is a USD file
  (`scene_generation/usd_tools/build_grid_storage_scene.py` writes `.scene_usd/grid_storage/scene.usda`);
  its drawer is `corner_nightstand_tier_2_drawer`. The house scene (`phygs/configs/scenes/house.yaml`)
  and its episode `dev_house_drawer` were added. The per-scene builders were removed;
  `episodes/scenes.py` keeps `builder` and `scene_config` for `phygs/ui/cli.py` until it
  uses `UsdScene` directly (2026-10-15).
- Episode harness, step 2: the scripted oracle agent (`phygs.agents.oracle`,
  strategy in `oracle.yaml`), which retrieves the target with privileged hints, and the robot
  tests `check_retrieve.py`, `check_drawer_stroke.py` and `check_drawer_place.py` built on it
  (shared start and thresholds: `episode_check.py`, `spot_episodes.yaml`).
  `Environment.truth()` gained `held()` and `inside()`. All four checks pass headless (2026-10-15).
- Episode harness, step 1 (design: `docs/design/episode_harness/index.html`): `examples/run_world.py`
  runs one episode from a spec (`configs/episodes/*.yaml`, `phygs.episodes.spec.EpisodeSpec`)
  with an agent `run(session)` (`phygs.agents`, e.g. `template`). The
  `phygs.episodes.environment.Environment` builds the scene from the registry
  (`episodes/scenes.py`; the grid storage scene, `phygs/configs/scenes/grid_storage.yaml`), owns the
  simulation loop on the main thread (charged or realtime clock), and writes `report.json` and
  `audit.jsonl`. Agents get an `phygs.episodes.session.AgentSession`: the skill
  clients their tier allows, observations, `wait`/`sleep`/`stream`, and privileged hints
  (`session.hints`) only for granted channels, every call logged. The judge
  (`phygs_robot_tests.tasks.retrieve.RetrieveJudge`) is a placeholder that reports
  `not_judged`. Verified headless: the template agent, and a table pick-and-stow through the
  session on the realtime clock with video (2026-10-15).
- Robot UI, phase 1 (watch only): `phygs/ui/cli.py` runs a headless episode and serves a
  local browser page (`phygs.ui`; design in `docs/design/ui/design.html`). It shows the
  stitched Front view (`phygs.ui.stitch.FrontStitcher`), the frontleft,
  frontright and hand cameras, the depth-built map, robot state, skill step lines, commands
  and a timeline, and has an E-stop. Only the agent view of the observation pipeline reaches
  the page (2026-10-14).
- Robot UI, phase 2: the page runs `navigation.navigate_to` (goal and heading picked on the
  map), Drive (`locomotion.command_velocity` renewed while a key is held) and `locomotion.stop`,
  and cancels running commands; rejected calls show the API's raw error. The UI episode
  navigates with the built-in planner on `ObservedDepthMap` (front-camera depth, declared flat
  floor) (2026-10-14).
- Robot UI, phase 3a: Arm joints (`manipulation.move_joints`), Joint path
  (`manipulation.execute_trajectory` through saved poses), Stow, and gripper open/close, with the
  direct skill backend; the depth and 3D point under the cursor in each camera view
  (`phygs.ui.targets`); a reorganised one-screen layout (2026-10-14).
- Robot UI, phase 3b: Look at, Move to pose, Move along, Pick and Place from a point clicked in a
  camera view (`phygs.ui.translate` turns choices into API arguments). The UI
  episode plans collision-free arm paths on the robot's own depth
  (`phygs.ui.arm_planning.OnboardPlanner`, MoveIt or cuRobo), for the skills and
  for `manipulation.move_to_pose`. The planner's map starts over after a pick or a release, so a
  picked-up object does not block its old place (2026-10-14).
- Robot UI, phase 4: the page shows the task until Start (the simulation waits), and ends with
  End episode, the time budget or E-stop. Each episode writes `report.json` (outcome, times, the
  judge's result, commands per skill, failures per code, Drive time) and a `recording` of every
  message, which `phygs/ui/replay_cli.py` plays back in the browser without Isaac Sim
  (`phygs.ui.recording`, `phygs.ui.replay`) (2026-10-15).
- Robot UI, phase 5: Open drawer and Close drawer from a handle bar clicked in a camera view
  (`phygs.ui.translate.handle_part` builds the drawer from the click's depth and the
  front's normal; `PerceivedParts` gives it to `manipulation.open`/`close` as their `parts`
  source, without simulator data). Plan preview for Arm joints and Move to pose: the page shows
  the planned gripper path and arm on the Front view and the map, and runs that very trajectory
  (`manipulation.execute_trajectory`). The arm planner's map starts over after a drawer skill,
  so a drawer that moved does not block the arm where it was (2026-10-15).
- `robot.manipulation.move_along_axis(axis, distance_m, *, end_pose=None)`: one skill for a
  straight wrist motion. Without `end_pose` it reaches back from the current pose; with
  `end_pose` (or candidates) it moves to the start pose `distance_m` back along `axis`, then
  straight to the end pose. It refuses with `out_of_reach` before moving when the start pose or
  the line is out of reach or would change the arm configuration (2026-10-13).
- `ArmKinematics` (`execution/manipulation/kinematics.py`): arm forward and multi-start inverse
  kinematics from the planning URDF, joint limits only; it returns the solution nearest the
  seed and keeps 1e-4 rad inside the limits; the skills use it (2026-10-13, 2026-10-14).
- `check_manipulation_planning.py --scene house --task episode --layout retrieve`: the
  episode on the house scene's kitchen island (2026-10-13).
- `check_manipulation_planning.py --task episode`: a full evaluated episode in the grid scene
  (navigation and manipulation on one runtime; open and close in two strokes; place into the
  drawer) (2026-10-12). Layout `retrieve` picks the target from the drawer and places it on
  the table (2026-10-12).
- Velocity command dead band for Spot (`LocomotionController.DeadBand`, `command_dead_band` in
  `spot_locomotion.yaml`): every velocity command within ±0.05 m/s (±0.02 rad/s) is sent as 0,
  and one below the lowest walking speed in its direction is raised to that speed; the
  `command_velocity` narration names the command sent (2026-10-13).
- Skill step narration: manipulation, navigation and locomotion skills log one line per
  step on `phygs.skills.<kind>`; `SkillFeedback.step` holds the current
  manipulation step; `SpotControl.narration` (`phygs.tools.narration`)
  prints each line with the episode time and keeps them for reports and video overlays.
  The manipulation robot test draws them on its video and adds `narration`, `base_track`
  and `drawer_contacts` to its report (2026-10-12).
- `robot.manipulation.stow()` (fold into the navigation transit posture) and
  `robot.manipulation.look_at(target)` (aim the hand camera); `SkillConfig.stow_posture`,
  `prestow_posture` and `look_camera` are required (2026-10-08).
- `look_at(target, *, distance_m=0.5, elevation_deg=45.0)`: the caller chooses the view;
  the skill moves to that one view and succeeds when the arm has settled (2026-10-09).
- Drawer skills `robot.manipulation.open(target, distance_m=None)` and
  `close(target, distance_m=None)` for sliding parts with a bar handle (object targets
  only, `ArticulatedPart` from `phygs.simulation.parts.sliding_part`);
  `pick(target, approach=None)` grasps along any downward or horizontal approach;
  `ManipulationClient.episode_id`; `--task drawer` in `check_manipulation_planning.py`
  (2026-10-08).
- Docs hub [docs/README.md](docs/README.md), visual pages in `docs/design/visual/`
  (status board, who-sees-what, frames and conventions), Mermaid diagrams in
  `ARCHITECTURE.md`, and this changelog (2026-10-06).
- Collision-aware reaching with MoveIt/OMPL or cuRobo V2 through
  `robot.manipulation.move_to_pose(pose)` with a configured pose planner; full-robot
  collider geometry, independent path validation, and the `require_observed` /
  `observed_obstacles` unknown-space policies (2026-10-04 to 2026-10-06).
- Grasp-and-lift reference agent: gripper holds a bounded squeeze after it stops on
  an object (2026-10-06).
- Basic manipulation in the shared runtime: `robot.manipulation.move_joints`,
  `move_to_pose`, `execute_trajectory`, `open_gripper`, `close_gripper`, `cancel`,
  with gravity feedforward and measured completion.
- Bear-house diagnostic, USD rigid-body registration for reset, and the optional
  frontier/coverage viewpoint selector `helpers/exploration.py` (2026-10-02).
- ForageBench scene fetch with pinned commit, checksum and offline scene records
  (`phygs/scenes/fetch.py`).
- Optional observation pipelines: `PipelineConfig`, `ServiceDeclaration`,
  `ProcessingService`, private service views and a `LocalizationEstimator` interface.
- First object-search task: `tests/robot_tests/tasks/object_search_api.py` and an independent
  RGB-D evidence evaluator.
- Observed-depth navigation map (`ObservedDepthMap`) for both navigation backends.
- Depth mapping helper `helpers/mapping.py` (`DepthMapper`).
- Optional Nav2 backend with grid, depth-map and `PointCloud2` inputs; measured
  transit envelope `phygs/configs/robots/spot_transit.json`; `check_nav2_pipeline.py`.
- Known-map point navigation: `robot.navigation.navigate_to`, A* planner, follower,
  and the point-navigation evaluator.
- Timestamped observations: `robot.observe()` with joints, IMU, calibrated RGB-D,
  explicit localization and permission filtering.
- Runtime-backed locomotion: `robot.locomotion.command_velocity`, `stop`, `feedback`.
- Shared episode runtime: `EpisodeRuntime` with integer clock, exclusive joint
  ownership, command expiry, reset, settling and emergency stop.

### Changed

- Component robot tests build the robot from the files episodes use (`common.robot_scene()`:
  the robot's `robot.yaml` and cameras); `locomotion.yaml`,
  `footprint.yaml` and `observations.yaml` keep only test settings (2026-10-28).
- `check_nav2_pipeline.py` runs only the house search (`check_house.py`); `--input`, `--task`,
  `--case`, `--manual`, `--moved_obstacle` and `--camera_view` are gone. Navigation is
  `episodes/check_navigation.py`, the moved obstacle `episodes/check_dynamic_obstacle.py`
  (2026-10-28).
- `tests/robot_tests/configs/navigation.yaml` holds the navigation agent test's cases.
  `phygs/execution/navigation/launch/nav2.launch.py` requires `experiment:=` (for example a run's
  `nav2_experiment.yaml`); its default was that file (2026-10-28).
- **Breaking:** `SpotControl` is `RobotControl`; `RobotControl.create(sim, scene, robot, ...)` takes the
  robot description and `manipulation=True` instead of `manipulation_config` (2026-10-28).
- **Breaking:** episode specs are schema 2: `robot: {model, start}`; `model` is a robot name or a
  `robot.yaml` path; the spawn height comes from the robot (`robot.height` is gone; the house episodes
  now spawn at 0.5 m instead of 0.6 m) (2026-10-28).
- **Breaking:** `phygs/configs/robots/spot.yaml` and `spot_transit.json` moved to
  `phygs/robots/spot/robot.yaml` (new schema) and `navigation/transit.json`; the
  policy files to `robots/spot/locomotion/` (`policy.pt`, `env.yaml`, `locomotion.yaml`). Camera
  prims, `skill_camera`, the fall height and the planner's cameras, volume and obstacle radius moved
  from `phygs/configs/environment.yaml` into the robot (camera roles, `planning`); `light` is new there
  (2026-10-28).
- **Breaking:** `robots/spot.py` split: `make_spot_scene_cfg` / `spot_simulation_cfg` are
  `robots.scene.make_scene_cfg(robot, ...)` / `simulation_cfg`; `spot_actuator_cfgs` and
  `SpotFlatTerrainPolicy` live in `robots/spot/` (the policy takes its settings path and `physics_dt`;
  `default_control_dt` is gone: the physics step is `phygs/configs/environment.yaml` `physics_dt`)
  (2026-10-28).
- **Breaking:** `planners.model.JOINTS`, `lowering.TOOL_FRAMES`, `session.ARM_JOINTS`,
  `front_view.FRONT_CAMERAS` and `ui.session.ARM_LINKS` are gone: read the robot description
  (`robot.arm.joints`, `robot.gripper.tools`, `session.robot`, `cameras_for("front")`).
  `prepare_model`, `collider_spheres`, `usd_only_boxes` and `audit_collision_geometry` take the
  robot (or its `planning` section); `NavigationConfig.arm_posture` defaults to none (navigation sets
  the robot's stow) (2026-10-28).
- `phygs/configs/navigation/spot_nav2.yaml` is `phygs/configs/navigation/nav2.yaml`; the launcher sets MPPI's
  motion model and the lattice from the base's kinematics (`base_kinematics`) (2026-10-28).
- Observations, the footprint and the Environment read the base pose from the base link
  (`simulation/base.py`), not the articulation root: the same for Spot (2026-10-28).
- **Breaking:** the gripper geometry is tool frames with one axis convention (x the approach, z the
  jaws' closing axis): `gripper.tools` gains `grasp` (required), `handle` and `fixed_jaw`, and
  replaces `gripper.grasp` (`grasp_height_m`, `grasp_center_z_m`, `fixed_jaw_z_m`,
  `top_down_wrist_xyzw`, `handle_depth_m`, `handle_jaw_z_m`). `RobotInfo.tools` maps each tool to
  its pose in the wrist link; `RobotInfo.grasp` is gone. The routines and the UI place the tool
  frames, so the Franka hand (approach along its z) works; Spot's routine poses are unchanged.
  The UI's Move to pose now puts the clicked point in the middle of Spot's jaw gap (the `grasp`
  frame, 2.9 cm along the closing axis from the old wrist axis) (2026-10-28).
- `PointNavigationTask.minimum_height` defaults to None (no height check); the robot tests pass the
  robot's `fallen_height` (Spot 0.35 m, was 0.25 m), and the evaluator samples the base link, so a
  base on planar joints is not judged fallen (2026-10-28).
- `RobotControl.create` starts the arm and the gripper at their configured start (`arm.stow`,
  `initial_joints`) and every reset returns there; component tests no longer start from the USD's
  joint positions (2026-10-28).
- `robot.yaml` `arm.motion` requires the limits and tolerances in joint units (`JOINT_MOTION`); the
  planners' joint velocity and acceleration limits come from it (MoveIt: half the acceleration)
  (2026-10-28).

- Episodes and the UI start with Nav2's map unknown beyond 1 m of the robot's start
  (`navigation.point_cloud.explored_radius`; `nav2_navigation(start=...)`); the cameras reveal it.
  Remove `explored_radius` to declare the whole floor free again (2026-10-27).
- The bridge's route check (`costmap_blocks_path`) no longer counts unknown cells (255) as blocked:
  Nav2 plans through them (2026-10-27).
- The robot-test routines stop within 4 cm of a stance (`routines/params.yaml` `goal_tolerance`,
  was 3 cm), `navigate_to`'s default; so does `agents/depth_example.py` (2026-10-27).
- `RenderCameras` renders cameras of the same size and data types in one tiled render product and
  cuts the tiles on the GPU; `tiled=False` keeps one render product per camera (2026-10-27).
- Nav2 runs beside the simulation: the bridge applies Nav2's newest velocity command instead of
  waiting for one each control period (`Nav2Config.command_grace` replaces `poll_timeout`), route
  checks and `costmap()` read the published `global_costmap/costmap_raw`, and the local costmap is
  a 6 m rolling window (2026-10-26).
- Nav2 maps the camera point clouds with its 3D `VoxelLayer` (`nav2.launch.py` `voxel_layer`):
  floor returns clear along their rays in 3D, so an obstacle that moves away clears and one a ray
  passes over stays. It replaces the `pointcloud_to_laserscan` scans of 2026-10-25 (the bridge's
  `<camera>_level` frames and the experiment's `scan.fov_deg` are gone) (2026-10-26).
- Smac allows routes through unknown cells (`allow_unknown: true`); maps without unknown space
  are unaffected (2026-10-26).
- `check_nav2_pipeline.py --input pointcloud` runs `check_dynamic_obstacle.py` (no
  `--dynamic_obstacle` flag); `--moved_obstacle` needs `--input pointcloud`. The house search runs
  `navigation.explore()` (2026-10-26).

- The robot's Gemini 2 cameras no longer see through near surfaces: `spot_model/spot_arm_w_cam.usd`
  overrode every `Stream_depth` prim's clipping range to (0.2, 20) m, so the renderer dropped
  anything nearer than 20 cm in RGB and depth (the hand camera saw through drawer fronts). The
  override is removed and NVIDIA's asset value (0.005, 1000) applies. The sensor's minimum range is
  now `ObservationConfig.depth_min` (0.2 m): nearer depth is 0 and invalid for every consumer
  (agents, skills, the arm planner, Nav2, the recorder and the UI); images show near surfaces. The
  footprint profile was measured again for the new USD hash (2026-10-25).
- The stitched front view moved from `phygs.ui.stitch` to
  `phygs.tools.front_view` (with `camera_frame` and `depth_colors`, which left
  `ui.codecs`), so the UI and the video share it; import from `helpers.front_view` (2026-10-25).
- `check_mapping.py --dynamic_obstacle` is now its own robot test, `check_dynamic_obstacle.py`
  (`check_nav2_pipeline.py --input pointcloud --dynamic_obstacle` runs it, so the `nav2_dynamic`
  suite entry is unchanged). The depth-mapping scene they share is `robot_tests/mapping_scene.py`
  (also used by `check_observations.py`), and the robot's ground-truth sample for the evaluators
  is `robot_tests/physical.py` (also used by `check_navigation.py`) (2026-10-23).
- Tighter robot-test limits (2026-10-23): `check_manipulation_planning.py` fails on any arm or
  held-object contact with the table or clutter above `contact_record_n` (0.5 N), naming the first
  (it was recorded only; 0 N in every passing run since 2026-10-21); `check_locomotion.py` allows a
  mean velocity error of 0.1 instead of 0.2 (at most 0.073 measured).
- The UI's Navigate to uses the public API's tolerance defaults (`NavigationGoal`, 0.04 m and
  0.04 rad) instead of its own 0.18 m and 0.12 rad (2026-10-23).
- `check_drawer_stroke.py` records the planned pull and requires at least
  `stroke_min_opening_m` (0.20 m, `configs/episodes.yaml`); before, a short planned pull passed
  (2026-10-23).
- Docs synced (2026-10-23): `PLANS.md` (the house episode is the small cardboard box and passes),
  `ARCHITECTURE.md` (skills are `stow` and `look_at`; the recorder and provenance modules),
  `docs/manipulation_skills.md` (removed straight-motion and recipe skills), the oracle page of
  `docs/design/episode_harness/` marked as history, and the implemented feedback in
  `docs/design/manipulation_contract/`.

- The episode report keeps the grasp apart from other arm contacts: `grasp_contacts` lists the
  jaw contacts (`manipulation.jaw_links` in `phygs/configs/robots/spot.yaml`) of a hold, from the close
  command to the opening (`ManipulationClient.gripper_closed`), with a body both jaws pinched in
  that hold, including the first jaw landing and a one-jaw pull (`episodes.recorder.Recorder`);
  `contacts` keeps the rest. A 150-170 N squeeze on the drawer
  handle or the picked box no longer reads as a collision. Readers of `report["contacts"]` that
  want the squeeze too read both lists (2026-10-23).

- Deployed a new Spot walking policy that observes the arm (`spot_policy.pt`, `spot_env.yaml`:
  60 inputs, the arm's 6 positions and velocities added; `SpotFlatTerrainPolicy` reads them from
  the policy's configuration). With the arm folded it is no worse than the tuned 2026-10-19 policy
  (the stand work's reference) on any stop, walking or navigation measure and reaches more
  navigation goals; with the arm swinging it
  tilts 2-3 instead of 8-11 deg (2026-10-22, `output_check/20261021-1859_h56_15seed`).
- The Spot stand uses PhysX joint drives (`stand_drive` in `spot_locomotion.yaml`, replacing
  `standing_gains`; `JointTargets.drive_stiffness` / `drive_damping`,
  `SpotLegActuator.set_physx_drive`) and switches once all feet are down (`stand_switch`: new
  `yaw_speed`, `feet_spread`, `feet_down_after`, `feet_down_spread`; latest switch 1.5 s, was 0.4 s).
- Stops slow the policy command at most at 2.0 / 1.0 m/s² and 2.0 rad/s²
  (`LocomotionController.Deceleration`, `command_deceleration`) and keep the legs for that ramp
  plus 1.6 s (`stop_stand_seconds`, was a fixed 0.8 s; `TimedStopController`). Build the
  controller with `LocomotionController.load(policy)`; the constructor now takes `deceleration`
  and `stand_seconds` (2026-10-22).
- Navigation's footprint is the measured standing outline instead of the walking sweep's circle
  (footprint profile schema 2 with `standing`, written by `check_footprint.py`). The built-in
  backend plans it as a rectangle plus 0.05 m padding in position and heading
  (`RectangleFootprint`, `transit_footprint`, `GridMap.with_footprint` / `GridMap.plan_route`;
  turns in place checked; near the start it may back away or sidestep); Nav2 plans the circle
  enclosing the outline (0.686 m). `KnownBoxMap`, `ObservedDepthMap` and `DeclaredFloorMap` take
  `footprint=` instead of `radius=`; `KnownBoxMap.physical_clearance` takes the heading
  (2026-10-22).
- Spot's articulation solves 8 TGS position iterations (was 4) and PhysX applies gravity in every
  position iteration (`spot_simulation_cfg`, used by every Spot simulation): standing on PhysX leg
  drives the arm sat 6-8 mrad off target and the robot reported 0.04 rad/s joint velocities while
  still. Now 3 mrad and none; stop wobble and settling time fell by half to 90%, no stop or
  navigation measure worse (15 seeds). The mapping check reads the map once the lifted obstacle
  is parked out of view, with its face in the camera's view (2026-10-22).
- Fine-approach moves at the lowest walking speed last as long as measured short moves need to
  cover the error (`FineApproach.Coverage`, `lowest_speed_moves` in `spot_locomotion.yaml`)
  instead of error / speed (2026-10-22).
- Re-measured the Spot command dead band for the deployed policy (`command_dead_band`): commands
  at or below 0.04 m/s (0.04 rad/s, was 0.05) are sent as 0, and larger ones up to 0.08 m/s
  (0.08 rad/s) are raised to 0.08 in every direction. Previously 0.05-0.07 m/s and 0.10 rad/s
  (2026-10-22).

- **Breaking:** robot tests are reorganised (design: `docs/design/archive/robot_tests_housekeeping/index.html`).
  `tests/robot_tests/episodes/` holds the episode tests (scripted plans through the agent session,
  judged with `environment.truth()`), `components/` the subsystem tests, `routines/` the plans'
  building blocks (was `agents/oracle.py`; strategy in `routines/params.yaml`), `configs/` the test
  settings (was `spot_*.yaml` next to the tests, e.g. `spot_mapping.yaml` -> `configs/mapping.yaml`).
  Every robot test takes `--output DIR` (required, fresh), runs automatic and headless by default
  (`--viz kit` to watch, `--manual` replaces running without `--auto`; `--auto` and `--report` are
  gone), writes `DIR/check.json` and prints `<test>: <status>; <DIR>` (`common.py`).
  `run_suite.py episodes|manipulation|base|nav2` runs a group. `check_footprint.py` writes
  `DIR/spot_transit.json` (was `--output FILE`); the Nav2 pipeline's simulation result is
  `DIR/simulation/check.json` (was `physical_report.json`) (2026-10-18).
- **Breaking:** the robot the product runs is `phygs/configs/robots/spot.yaml` (robot, scene,
  manipulation, skills, a new `gripper:` section, navigation limits, footprint profile), read by
  the episode harness (`environment.yaml` `robot_config`; `navigation_config` is gone) and the UI
  (`phygs/configs/ui/episode.yaml` `robot_config`, was `manipulation_config`). `SkillConfig` keeps
  `backend`, `move_seconds`, `stow_posture`, `prestow_posture` and `look_camera`; read grasp and
  handle geometry from `gripper:` instead. The UI's assumed held box is
  `phygs/configs/ui/episode.yaml` `held_object_size_m` (2026-10-18).
- `check_manipulation_planning.py` runs the table task only (MoveIt or cuRobo, known boxes or
  depth); its episode task and house scene are covered by the episode tests on
  `dev_grid_drawer`, `dev_house_drawer` and `dev_grid_table` (2026-10-18).
- Robot tests read trusted diagnostics through environment-only properties instead of private
  members: `NavigationClient.backend` (e.g. `backend.feedback(id)` for the planned path, Nav2's
  `costmap_cost`), `LocomotionClient.controller` (its `policy`), `ManipulationClient.controller`
  (mode, reason, reference) and `RuntimePipeline.cameras`; agents cannot reach them (2026-10-18).

- The oracle (`agents/oracle.py`) composes the pick, the place and the drawer strokes itself from
  `move_to_pose` (planned, to where a straight motion starts), straight end-effector lines
  (`execute_trajectory` with an `EndEffectorTrajectory`) and the gripper commands, and passes the
  held object to every planned move (`oracle.pick` returns it). It reads the target hint again after
  opening a drawer: the drawer moves the object (2026-10-18).
- Robot UI: Move along (skill `move_along`, was `move_along_axis`) runs a straight wrist line
  through `manipulation.execute_trajectory`, from where the wrist is, at 0.1 m/s
  (`ui.translate.line`). While the gripper holds something, the UI's arm planner models it as
  the assumed `held_object_size_m` box between the jaws (2026-10-18).
- **Breaking:** `move_to_pose` takes a **hand** pose (between the jaws, `tool="hand"`) by default;
  pass `tool="wrist"` for a wrist link pose as before. It takes `held=HeldObject(...)` (box in the
  wrist frame, scene `name`): the planner avoids obstacles with the carried object and leaves it
  out of them, instead of reading a held object set by a skill. `HeldObject` gained `name`
  (2026-10-17).
- **Breaking:** the project runs at 250 Hz physics: `physics_rate_hz: 250` in
  `execution/locomotion/spot_locomotion.yaml` and `physics_dt: 0.004` in
  `configs/episodes/environment.yaml` (were 500 Hz / 0.002 s). The policy keeps its 50 Hz period
  (decimation 5); standing gains and the transit footprint are the 250 Hz ones. Set 500 / 0.002 to
  cross-check a 250 Hz result (2026-10-17).
- A joint move (`move_joints`, `stow`, gripper commands' arm) is tracked while it runs (every arm
  joint within `tracking_tolerance`, 0.15 rad, of its moving reference) and ends when the arm
  stops, as trajectories do; it no longer needs every arm joint within 0.03 rad of its target, which
  a held object's load prevents at 250 Hz (a stow with an object timed out). A commanded gripper
  still has to reach its target or stop on an object. `execute_trajectory` refuses a plan whose
  first point is more than `tracking_tolerance` from the measured arm (was 0.03 rad) (2026-10-17).
- A trajectory (`execute_trajectory`, and every planned or straight skill motion) ends when the
  arm stops; tracking is still checked while it runs, but the arm is no longer compared with the
  last sample (0.03 rad). `move_along_axis` judged its line instead, against the measured start
  pose: within 1 cm of the line (2 cm while holding an object) and 0.05 rad of the start
  orientation, else `execution_stalled` (removed 2026-10-18 with the skill; see Removed).
  The jaws on a drawer handle held the wrist roll 0.039 rad off the last sample at 250 Hz and
  every drawer pull timed out (2026-10-16).
- **Breaking:** the default `navigate_to` goal tolerance (and `NavigationGoal`'s, which the point
  navigation task also judges by) is 0.04 m / 0.04 rad, was 0.18 m / 0.12 rad; Nav2's
  `goal_checker` in `phygs/configs/navigation/spot_nav2.yaml` matches. Pass
  `position_tolerance=0.18, heading_tolerance=0.12` to keep the old acceptance region; an agent
  that drives with `command_velocity` needs it, as it has no fine approach (2026-10-16).
- The deployed Spot locomotion policy (`execution/locomotion/spot_policy.pt`,
  `spot_env.yaml`) was replaced with one trained on flat ground with the arm in library poses
  (`scripts/reinforcement_learning/spot_arm_locomotion`). It tracks 0.1 m/s and 0.2 rad/s in every
  direction (75-119 %), with the arm stowed, held or moving; the old one did not move below
  0.10-0.26 m/s. The command dead band (`command_dead_band` in `spot_locomotion.yaml`) is now
  0.05-0.07 m/s and 0.10 rad/s (was 0.10-0.26 m/s and 0.02-0.04 rad/s), and the navigation
  footprint `phygs/configs/robots/spot_transit.json` was measured again. At 250 Hz physics the policy
  fell during the footprint measurement, so `spot_transit_250hz.json` is still the previous
  policy's and navigation at 250 Hz refuses to start (2026-10-16).
- The Spot policy wrapper (`SpotFlatTerrainPolicy`) switches to standing after a zero command once
  the base is nearly still and no leg joint moves fast, or after 0.4 s at the latest
  (`stand_switch` in `spot_locomotion.yaml`), and holds the legs where they are instead of jumping
  to the default posture. A stop from 0.1 m/s moved 3-6 cm and turned 0.14-0.48 rad; now
  1.2-2.2 cm and at most 0.014 rad (2026-10-16).
- Spot's actuators default to two groups instead of four: `robots.actuators.SpotLegActuator`
  for the twelve legs (hips as `DelayedPDActuator`, knees as `RemotizedPDActuator`, separate
  random delays) and one implicit group for the arm and gripper. Torques are identical to
  IsaacLab's classes for every delay (`tests/test_spot_actuators.py`); the actuator write takes
  0.61 ms per physics step instead of 1.19 ms. `robots.spot.spot_actuator_cfgs("isaaclab")`, or
  `robot.actuator_groups: isaaclab`, keeps IsaacLab's layout, for example for Newton-native
  actuators (2026-10-16). **Breaking:** the groups are named `spot_legs` and `arm_and_gripper`;
  code that addressed `spot_hip`, `spot_knee`, `arm_hold` or `gripper_hold` sets
  `actuator_groups: isaaclab` or uses the new names.

- Observation cameras render through `simulation.cameras.RenderCameras` instead of IsaacLab
  camera sensors: each camera's RGB and depth come from Replicator annotators on the GPU and are
  copied once to the scene device, the pose is the parent body's physics pose times the fixed
  mount from USD, and Kit's
  hidden 1280x720 viewport is no longer rendered headless. Images are identical to IsaacLab's
  camera output (RGB and depth difference 0 over 157 frames, three cameras). With three
  640x480 cameras and Spot walking in the Bear house, RTF rose from 0.33 to 0.55 (2026-10-16).
  **Breaking:** `robots.spot.make_spot_scene_cfg` builds cameras as
  `simulation.cameras.RenderCameraCfg` scene entries, so `scene[name]` is no longer an IsaacLab
  `Camera`; read images from observation packets. `IsaacLabObservations` takes `cameras=` (a
  `RenderCameras`, built by default) instead of `render=`; its `cameras` property lets a second
  adapter on the same scene share them.

- The handle-grasp keys `handle_depth_m` and `handle_jaw_z_m` moved from `manipulation.skills` to
  `manipulation` in `tests/robot_tests/spot_manipulation.yaml`; agents and tests that grasp a bar
  handle read them there. The oracle keeps its own copy in `oracle.yaml` (2026-10-16).

- Headless runs with cameras (no GUI or visualizer) have PhysX write body poses to Fabric,
  which only rendering reads, only in the two physics steps before each camera capture and from
  restore through settling (`simulation.fabric.FabricTransformWrites`, created by
  `SpotControl.create`; `IsaacLabWorld` and `IsaacLabObservations` take it as `fabric_writes`).
  PhysX wrote 310 house bodies every step, about 1.3 ms per step (2026-10-15). A `sim.render()`
  between captures shows the poses of the last capture.

- `register_scene_bodies` registers a scene's own rigid bodies as one `RigidObjectCollection`
  (`scene_assets.SCENE_BODIES`), each named by its USD path, instead of one `RigidObject` per
  body named `scene_body_NNNN`: IsaacLab cleared every asset's force buffers every physics tick,
  3.6 ms per tick for the house's 310 bodies. PhysX simulates the same bodies; with a drop and a
  drawer pull, 307 of 310 bodies matched the per-body registration within 3 um (the pulled
  drawer's contents differ within their sensitivity to 0.01 % of the force). Read and write them
  with `scene_assets.SceneBodies` (`pose`, `velocity`, `write_pose`, `write_velocity`,
  `hold_force`, `body(path)` for `RigidObject`-like reads); `IsaacLabWorld` restores them at
  reset and applies held forces to the named bodies only; `BoxGeometry.scene_key` may name a
  scene body by its path (2026-10-15). **Breaking:** code reading `scene["scene_body_NNNN"]`
  reads `SceneBodies(scene).body(path)` instead.

- The house storage-unit helpers moved from `tests/robot_tests/house_storage.py` to
  `phygs.simulation.storage_units`; the old module re-exports them until its users
  (`check_manipulation_planning.py`, `phygs/ui/cli.py`) import the new one (2026-10-15).
- Robot UI: `phygs/ui/cli.py` runs an episode spec (`--spec`, default
  `phygs/configs/ui/grid_drawer.yaml`; docs/episode_harness): the scene comes from the scene registry
  (`grid_storage`), the task, tier and budget from the spec, and `tasks.retrieve.RetrieveJudge`
  scores it. The page offers only the skills the spec's actions allow. The UI's own grid scene
  is gone; use `--spec` in place of `--scene grid|house` (`phygs/configs/ui/house_drawer.yaml` for the
  house). Scenes load with `phygs.simulation.scene.UsdScene`. The UI imports storage units from `phygs.simulation.storage_units`
  (2026-10-15).

- `pick` centres the object in the middle of the jaw gap (`SkillConfig.grasp_center_z_m`, 0.029 m
  in the wrist frame: the fixed jaw's pad and the moving jaw opened 75 deg), not on the gripper
  axis, which only fitted objects under ~7 cm wide (2026-10-14), and records the held object
  resting on the fixed jaw (`SkillConfig.fixed_jaw_z_m`, -0.036 m), where the closing jaw pushes
  it (2026-10-15). `SkillConfig` requires both.
- The house episode uses the kitchen island's bottom drawer and its 9.5 cm box: the top
  drawer's bottle, picked at 60 deg, did not fit the stow (2026-10-14).
- The episode tests open and close the drawer in one 0.35 m stroke from one stance, composed
  from `move_along_axis` and the gripper commands: a pitched handle grasp to open, a push with
  the closed gripper to close; their goal tolerance is 0.08 m (2026-10-13, 2026-10-14).
- `NavigationConfig.timeout` defaults to None: `navigate_to` has no time budget unless the
  caller passes `timeout` or a configuration sets one (2026-10-13).
- Robot tests record contact (`contact_record_n`, `furniture_contacts`, `held_contacts`) and
  base shift instead of failing on them; they judge the task outcome. The blocked drawer case
  expects the drawer stopped and the robot standing, not a particular failure (2026-10-13).
- A handle release (`open`, `close`) backs out 6 cm and then lifts 8 cm
  (`HANDLE_LIFT_M`): the next plan must not start in collision with the bar in the
  planners' sphere model of the jaws (2026-10-12).
- The manipulation robot test's `look_views` are lists tried in order (the agent's
  choice, not the skill's); drawer layouts may set `close_distance` (2026-10-12).
- The drawer robot test uses the top drawer of the house scene's two-drawer
  `corner_nightstand` (`tests/robot_tests/house_storage.py`) instead of the
  `tests/robot_tests/storage.py` unit, which was removed; `storage`/`drawer_layouts` in
  `spot_manipulation_planning.yaml` became `storage_unit`/`drawer_layouts`, and `look_views`
  sets the requested views (2026-10-09).
- `stow()` while holding an object carries it in the stow instead of failing with
  `holding_object`; the gripper closes only when empty (2026-10-09).
- **Breaking:** `SkillConfig` requires `handle_depth_m`, `handle_jaw_z_m` and
  `line_speed_m_s`. Add them to the `manipulation.skills` YAML (see
  `tests/robot_tests/spot_manipulation.yaml`) (2026-10-08).
- The planning check's independent guard uses 2 cm voxels (`map.resolution`), as fine as
  the planners' maps; 5 cm cells rejected paths the planner kept 1 cm clear (2026-10-08).
- Manipulation stillness (target reached, gripper stopped on an object, physical stop)
  uses the net joint change over `MotionConfig.motion_window_seconds` (0.05 s), not one
  tick: a jaw squeezing a fixed handle vibrates in place (2026-10-08).
- Spot USD: the 10 editor collision spheres on the gripper finger are disabled; the
  physics finger now matches its mesh. `phygs/configs/robots/spot_transit.json` was
  recalibrated (30 shapes, 1.243 m × 0.892 m) (2026-10-08).
- **Breaking:** the public APIs moved to the shared runtime. Use `SpotControl.create(...)`
  and `robot.<skill>` clients. The Spot-SDK facade, AO-Grasp and CuRobo runner are
  deferred and not imported by the public APIs.
- Locomotion execution moved to `phygs/execution/locomotion/`;
  policy files use the `spot_` prefix.
- IsaacLab launch scripts moved from `tests/isaaclab_test/` to `tests/robot_tests/`.
  Physical checks are `check_*.py`, manual by default, `--auto` for assertions.
- Scene generation, Blender and catalog helpers moved from `scripts/` to
  `scene_generation/usd_tools/legacy/`.
- The manipulation planning check uses Spot's onboard `frontleft`, `frontright` and
  `hand` cameras.
- `ARCHITECTURE.md`, `PLANS.md` and `README.md` reorganized: current system first,
  deferred stack in its own section. `old_README.md` moved to
  `docs/legacy_scene_generation_notes.md` (2026-10-06).

### Removed

- Robot tests replaced by agent tests (2026-10-28): `components/check_navigation.py` and
  `check_house.py`'s navigation task (use `episodes/check_navigation.py`),
  `components/check_dynamic_obstacle.py`, `mapping_scene.py` and `configs/mapping.yaml` (use
  `episodes/check_dynamic_obstacle.py`), `components/check_manipulation_planning.py` and
  `configs/manipulation_planning.yaml` (use `episodes/check_arm_planning.py` and
  `episodes/check_pick_place.py`).
- Housekeeping (2026-10-27): the unused scripts `phygs_scene_robot_test.py`,
  `spot_articulation_action_test.py` and `helpers/physics_optimizer.py`; the built-in A* left in
  `GridMap` (`plan`, `plan_route`, `segment_cost`, `traversal_cost`, `InvalidGoal`,
  `UnreachableGoal`) and `KnownBoxMap`'s clearance costs: Nav2 plans; the grid keeps the pose, turn
  and segment checks the fine approach uses. The 2026-10-18 robot test housekeeping page moved to
  `docs/design/archive/`.

- The framework's depth mapping: `helpers/mapping.py` (`DepthMapper`, `MappingConfig`,
  `MappingSnapshot`), `ObservedDepthMap`, `helpers/exploration.py` (`FrontierExplorer`),
  `check_mapping.py`, `check_nav2_pipeline.py --input depth_map`, `--coverage_only` and the suite's
  `nav2_depth_map` and `nav2_pointcloud` entries. Use Nav2's costmap (`backend.costmap()`) and
  `navigation.explore()`; an agent's own map comes from its camera stream (2026-10-26).

- `ObserveRequest`, `ObserveResult`, the planners' `observe()` and `PlannerProcess.observe()`: send
  the depth with `PlanRequest.images`; each plan builds its own map. `ui/arm_planning.py`
  (`OnboardPlanner`, `DepthFeed`) and `episodes/arm_planner.py`: use
  `execution/manipulation/arm_planner.ArmPlanner`. The UI planner no longer restarts after the base
  moved (each request carries the legs). `helpers/collision_map.robot_voxels` (no caller left).
  `check_manipulation_planning.py` lost its own depth masking (`RobotMap`), worker host and the
  depth-mode `scan` (the pick looks at the target first); its config's `map` and `validation`
  became `planner` (2026-10-23).
- The planners' `grasp(GraspRequest)` (`GraspRequest`, `GraspResult`), MoveIt's Task
  Constructor node and planner, and the `ros-jazzy-moveit-task-constructor-core` dependency. Nothing
  called them since the skills plan to a start pose and move straight (2026-10-13). Use
  `plan(PlanRequest)` (2026-10-23).
- `JointTrajectory.validate_speed` (no caller; execution slows a fast plan with `slowed()`),
  `OnboardPlanner.forget()` (its caller, the UI drawer tiles, was removed) and the `enabled`
  argument of the built-in navigation backend (`NavigationClient` enforces the task's
  permission) (2026-10-23).
- Robot-test leftovers: the table task's `probe` box, the unread `tracking_error_limit_rad`, the
  house search's `spawn_position`/`initial_free_radius` override (equal to the robot section's)
  and the unread `scene.env_spacing` keys (2026-10-23).
- `tests/test_local_physics_engine.py` (no tests) and two tests that checked nothing:
  `test_point_navigation_evaluator_does_not_require_agent_localization` and
  `test_the_retrieve_judge_is_a_placeholder_for_now` (2026-10-23).
- **Breaking:** 500 Hz physics support: the 500 Hz `stand_drive` gains (`spot_locomotion.yaml`;
  `SpotFlatTerrainPolicy` now rejects 0.002 s), the per-rate footprint lookup
  (`footprint.footprint_path_for_rate`, the `physics_dt` argument of `load_footprint`) and
  `phygs/configs/robots/spot_transit_250hz.json`. The project runs at 250 Hz only;
  `phygs/configs/robots/spot_transit.json` is the 250 Hz profile, re-measured with 8 solver position
  iterations (`output_check/20261022-1917_footprint_solver8`: standing outline 1.20 m x 0.52 m,
  Nav2 circle 0.660 m) (2026-10-22).

- **Breaking:** the oracle agent (`phygs.agents.oracle`, `oracle.yaml`,
  `run_world.py --agent oracle`). Its routines are `tests/robot_tests/routines/`; the retrieve
  plan is `plan(session)` in `tests/robot_tests/episodes/check_retrieve.py` (2026-10-18).
- `NavigationClient._diagnostics(id)`; use `NavigationClient.backend.feedback(id)` (2026-10-18).
- `tests/robot_tests/episode_check.py` (now `common.py` and `episodes/episode.py`),
  `tests/robot_tests/house_storage.py` (use `phygs.simulation.storage_units`) and
  `check_manipulation_planning.py --task`, `--scene` and `--backend none` (2026-10-18).

- **Breaking:** the skills `robot.manipulation.pick(target, *, approach=None)`, `place(target)` and
  `move_along_axis(axis, distance_m, *, end_pose=None)`, their backend code and the skills'
  straight-line check (`LINE_TOLERANCE_*`; the end-effector tracking of `execute_trajectory`
  replaces it). Compose them: open the gripper, `move_to_pose` to the start of a straight
  approach, a straight `execute_trajectory(EndEffectorTrajectory(...))`, close or open the
  gripper, and a straight line back out, as `oracle.pick`, `oracle.place`,
  `oracle.approach_straight` and `oracle.move_straight` do. `stow` and `look_at` stay skills
  (2026-10-18).
- **Breaking:** the skills' held-object state: `ManipulationClient.held_object`,
  `DirectSkills.held_object` and the `held_object` argument of the episode arm planner
  (`episodes/arm_planner.py`). Planners no longer fall back to an object a skill picked; pass
  `held=HeldObject(size, wrist_from_object, name)` to `move_to_pose`, `stow` and `look_at` while
  carrying (2026-10-18).
- Robot UI: the Pick and Place tiles, `ui.translate.pick_approach` and `translate.APPROACHES`.
  Close the gripper on an object and use Move to pose and Move along instead (2026-10-18).
- The drawer skills `robot.manipulation.open(target, *, distance_m=None)` and
  `close(target, *, distance_m=None)` (never released), their backend code and the `parts`
  argument of `create_skills` and `DirectSkills`. Compose a drawer stroke from `open_gripper`,
  `move_along_axis` (approach with `end_pose` at the handle grasp pose, then along the slide),
  `close_gripper` and a release, as `oracle.open_drawer` / `oracle.close_drawer` do (2026-10-16).
- `ArticulatedPart.handle_box()`; `ArticulatedPart` stays for scene hints and the oracle (2026-10-16).
- `SkillConfig.handle_depth_m` and `SkillConfig.handle_jaw_z_m`. Read the keys from the
  `manipulation` section of `spot_manipulation.yaml` (see Changed) (2026-10-16).
- `check_manipulation_planning.py --task drawer`, its `drawer_layouts` (`center`, `left`, `right`,
  `far`, `half_open`, `blocked`), `map.drawer_bounds`, the drawer checks and scene, and the stop
  block. `--task` is `table` or `episode`; drawers are covered by `check_drawer_stroke.py` and the
  episode tests (2026-10-16).
- Robot UI: the Open drawer and Close drawer tiles, `ui.translate.handle_part`,
  `translate.PerceivedParts`, `translate.BARS`, the `parts`, `handle_size` and `scene_moved`
  arguments of `UiSession`, `drawer_handle` in `phygs/configs/ui/episode.yaml`, and `face_normal` /
  `proud_center` in `ui/targets.py`. The tile group "OBJECTS AND DRAWERS" is now "OBJECTS" (Pick,
  Place). `tests/test_ui_drawers.py` became `tests/test_ui_previews.py` (plan previews only)
  (2026-10-16).
- The standalone Isaac Sim demos of 2026-09-09: `phygs/skills/manipulation/following_obj_rmf.py`,
  `pick_place_ori.py`, `proposed_methd/` and `utils/`. Use the `robot.manipulation` skills (2026-10-16).

- Robot UI: Declare target (`phygs.ui.search`) and the house search cube: retrieve
  is the only task (docs/episode_harness). `ObjectSearchEvaluator` stays for `check_search.py`
  (2026-10-15).

- The trajectory guard: `JointTrajectory(..., guard=)`, its controller hook,
  `execution/manipulation/validation.py` and the test's guard and moved-probe check. Nothing
  checks or stops a planned motion while it runs (2026-10-13).
- The pull's "hand off the line" stop in `open`/`close` (`LINE_DEVIATION_M`) (2026-10-13).
- The skills' grasp-planner path (`GraspPlanner`, `GraspPlan`, `create_skills(grasp_planner=)`):
  `pick` and `place` approach like `move_along_axis` too, planned to the start pose around every
  obstacle, the object included; a refused plan is `planning_failed` (was `no_grasp` /
  `no_place`) (2026-10-14).
- The handle contact region (`contact_box`, `HANDLE_CONTACT_MARGIN_M`, `HANDLE_CONTACT_DEPTH_M`):
  `open`/`close` approach the handle like `move_along_axis`, planned to the start pose around
  every obstacle, then straight in; the planner can no longer sweep the jaws through the bar
  (2026-10-14).
- The built-in navigation follower's dead-band lift, half-tolerance stop and near-goal heading
  hold; the dead band is handled in locomotion (2026-10-13).
- `SkillConfig.retries` and the skills' internal retries (pick, open, close) and the
  `look_at` view fallback: a skill makes one attempt and reports its failure; the agent
  decides whether to try again. Delete `retries` from skill YAML (2026-10-09).
- The Spot-SDK-shaped facade (`phygs/skills/spot`, `skills/core.py`, `skills/base.py`,
  and the SDK names exported by `skills`), the AO-Grasp client, sidecars (`docker/aograsp`,
  `docker/compose.aograsp.yml`) and viewers, the CuRobo runner (`manipulation_skill.py`,
  `curobo_interfaces.py`), `helpers/cam_utils.py`, `helpers/pcd_utils.py`,
  `scripts/phygs.py`, their configs and unit tests. Use `SpotControl.create(...)`
  and the `robot.manipulation` skills with MoveIt or cuRobo V2 planning (2026-10-08).
- The fixed-base AO-Grasp/CuRobo drawer smoke tests `tests/robot_tests/spot_manipulation_drawer.py`,
  `spot_manipulation_drawer_known_grasp.py`, their YAML, and the unit test that only read
  their source. Use `check_manipulation_planning.py --task drawer` (2026-10-08).
- Virtual floating cameras and the sphere robot self-filter from the manipulation
  planning check. Set sensor range on the USD camera prims instead.
- Experimental visibility and containment probes from manipulation planning.

### Fixed

- `--record` on robot tests without observations (locomotion, footprint) failed with "No
  observation configuration was supplied"; they record the scene view (2026-10-28).
- The Nav2 bridge no longer fails bringup when a costmap's activation blocks its node for longer
  than `service_timeout`: while waiting for the planner and controller to become active (up to
  `Nav2Config.bringup_timeout`, 60 s) it re-sends the clock, transforms and odometry the costmap
  needs and retries unanswered state requests instead of poisoning the transport
  (`nav2_known_map` startup race, 2 of the last 3 full suites; 2026-10-23). When Nav2's lifecycle
  manager stalls (a lost configure response, `nav2_depth_map` 2026-10-23), a node unchanged for
  `Nav2Config.stalled_transition_seconds` (15 s) is driven through its transition by the bridge.
- `navigate_to` fails after `NavigationConfig.max_goal_corrections` (3) route corrections when the
  settled pose stays outside the goal tolerances, in both backends, instead of replanning until the
  budget; the house search check asks for a 0.12 rad heading tolerance (`exploration.heading_tolerance`)
  with its 0.18 m arrival tolerance, since 0.04 rad is below what Nav2 settles to without the fine
  approach (`house_search` looped 120 s on a viewpoint 7 cm from the start, 2026-10-23).
- Robot UI: `UiServer.stop()` closes the page's connection first and no longer raises when a page
  is still connected (a `TimeoutError` there hid why a run ended). The WebSocket no longer
  compresses frames, which also removes an asyncio error when a page disconnects mid-frame
  (2026-10-23).

- Unit tests (2026-10-23): `test_nav2_transport.py` failed since the rectangle footprint
  (f417229) and skipped itself without ROS, hiding it (now skipped at module level). Tests that
  could not fail now can: the replay test checks the replay server's bridge, the pixel probe
  test probes an older frame, the drawer-ceiling test checks a point in the drawer above, and
  the handle grasp test checks the grasp's geometry instead of a copy of the code. The UI fake
  arm records the tool it is given. Fake worlds step at 0.004 s (250 Hz) instead of 0.002 s,
  and assertions read calibration values from the configuration files instead of copying them.
- `ManipulationClient.gripper_closed` stays true while squeezing a wide object whose stop angle is
  nearer the open than the closed angle (the 9.5 cm house box at -0.93 rad); the episode recorder
  sorted that hold's jaw contacts as unintended `contacts` instead of `grasp_contacts` (2026-10-23).
- The episode report's `provenance` is built for every spec; it was built only with the
  `navigation` action, so a `joints`-tier episode failed when writing its report (2026-10-23).

- An arm command no longer ends the episode when the simulator reports a joint resting slightly
  past its limit (a closed gripper at +1.8e-7 rad over its 0 rad limit made `Move along` raise
  `ValueError` in `ManipulationController`): measured joints held as targets are clamped to the
  limits, and a servo or trajectory reference outside the limits fails the command
  (`failed`, the limit in the reason) instead of raising (2026-10-23).

- Nav2 runs no longer swing about the route: RPP turns in place only above 45 deg and at 0.8 rad/s
  (`phygs/configs/navigation/spot_nav2.yaml`), so the locomotion deceleration limit no longer carries
  each turn past the threshold. The mapping test's navigation goals take
  `diagnostic.task_time_budget` (60 s; was a fixed 30 s): Nav2's detour around the dynamic
  obstacle arrives up to 1.8 rad off the goal heading and the 4 cm finish takes up to 20 s (2026-10-22).

- A gripper that closes on an object squeezes a bounded amount past the first contact, once, and
  within its joint limits. Grasp detection ran on every tick, so a jaw creeping on a soft or
  slipping object pushed the squeeze target further at each pause, up to past the closed end stop,
  and the out-of-range target stopped the episode (the house retrieve, 2026-10-18) (2026-10-18).
- Joint stillness is the change of the mean position between the halves of the motion window
  (`MotionConfig.motion_window_seconds`), not between its two ends. A jaw flipping between two
  contact positions every tick (5.6 mrad at 250 Hz) read as 0.11 rad/s whenever the window spanned
  an odd number of ticks, so a gripper close on an object timed out (2 of 6 planning-depth runs,
  `output_check/20261017-2301_depth_timeout_trace`) (2026-10-18).

- An arm that still moves 2 s after a stop fails its manipulation command ("did not physically
  stop", `execution_stalled`) instead of raising inside the controller and ending the episode
  (2026-10-17).
- After a successful arm motion the arm keeps its commanded joint targets
  (`ManipulationController.arm_hold`, `hold_targets`): when released, while a plan is computed and
  as the start of the next command. It held its measured joints, so a held object's load pushed it
  further each time; at 250 Hz a stow planned for 5 s with an object was refused ("Trajectory start
  does not match measured joints"). A failed or cancelled motion still holds where it stopped
  (2026-10-17).
- After a stop, the legs keep the policy's standing targets when locomotion releases them
  (`LocomotionController.hold_targets`, `SpotFlatTerrainPolicy.stand_pose`), not their measured
  positions: under load those sit below the targets, and re-holding them sank the base after each
  stop (median 1.7 cm at 250 Hz, 0.8 cm at 500 Hz) and tipped it after some left turns at 250 Hz
  (2026-10-16).
- `robot_tests/house_storage.copy_unit` keeps the house's solver iterations; the rebuilt house
  scene no longer overrides velocity iterations (2026-10-12).
- Straight-line arm motions (`open`, `close`) report where they leave the arm's reach and
  whether IK had no solution or jumped branches (2026-10-12).
- The robot collision-geometry identity (`CollisionFootprint.geometry_sha256`) no longer
  depends on where the robot stands: its link-frame bounds are rounded to 1 nm before hashing
  (placement changed them by ~1e-16 m). `phygs/configs/robots/spot_transit.json` carries the new
  identity of the same geometry (2026-10-12).
- The manipulation robot test clips known obstacle boxes to the parts outside the region a
  skill may touch, instead of dropping every box centred in it: a house drawer's front
  (centred just behind its handle) vanished from planning and the guard (2026-10-09).
- MoveIt times paths at 1.5 rad/s² instead of 3.0: faster swings swayed the legged base
  past the trajectory guard (2026-10-09).
- MoveIt checks the arm with the trajectory guard's spheres (`spot_planning_moveit.urdf`)
  instead of its meshes, which the spheres exceed by up to 5 cm: paths MoveIt accepted
  near furniture no longer fail the guard. MoveIt grasp approaches use the plan budget
  instead of MTC's 1 s default (2026-10-09).
- The cuRobo sphere-model cache key now includes the self-collision skip list; a changed
  list used to load a stale collision matrix (2026-10-08).
- `check_navigation.py --backend nav2` crashed at Kit start (exit 139): its ROS check
  imported NumPy before Kit. `require_ros2` now lives in
  `navigation/backends/ros2_check.py` (still importable from `backends.ros2`) (2026-10-08).

## 2026-05-06 and earlier

Summarized from git history.

### 2026-04 to 2026-05

- Added cuRobo Docker patch and installation instructions; cuRobo IsaacLab test.
- Added the manipulation skills stack (AO-Grasp client, CuRobo planner, drawer smoke test).
- Added the locomotion API, trained Spot-arm locomotion policy, and WASD test.
- Updated the Spot USD: foot links and contact, max joint velocity, no leg self-collision, camera poses.

### 2025-07 to 2026-04

- Added Docker setup and README.
- Added side and rear cameras for Spot; fixed collision and rigid bodies.
- Refactored the skill API and manipulation API; added the framework and basic integration (2025-12).
- Added the Spot + arm + depth extraction pipeline (2025-10).
- Updated Infinigen for the repo; articulated objects in Isaac Sim; initial project files (2025-05 to 2025-08).
