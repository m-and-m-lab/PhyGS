# Robot tests: the ladder

Each robot test spawns a scene and has the robot do one thing: walk, plan an arm move, pick and
place, open a drawer, search, retrieve. A test does three jobs at once:

- **Show.** `--record` writes a video of the run; `--viz kit` opens the simulator window.
- **Check.** It measures named values against thresholds and writes `check.json`, so a person or a
  coding agent can see what broke without watching.
- **Teach.** Its code is the reference for using that part of the framework.

The tests form a ladder in the order the framework is built. After a change, run the ladder up to
the rung you touched; the first failing rung is the lowest broken module.

## Two shapes

A test is written the way the real user of its subject would write it.

- **Agent tests** (`episodes/`) cover what an agent asks for: navigation, arm planning, pick and
  place, drawers, search, retrieve. A test is an episode spec (`configs/episodes/`), a plan that
  drives the robot through the agent session only, and checks on the other side: ground truth
  after the run (`world.truth()`) and, during it, a per-tick call
  (`World.run(during=...)`) that samples ground truth or moves a scene body. The plan names
  its steps (`episode.Steps`) so the test knows what it measures. Shared plan pieces are in
  `routines/`; a plan here is what an agent author copies.
- **Component tests** (`components/`) cover the machinery that carries out an agent's requests:
  the stage and reset, the cameras, the gait, the footprint, the arm's joint controller. Their
  reader works on that machinery, so they drive the modules directly, and some have keyboard
  control (`--manual`). They build the robot from the files episodes use (`common.robot_scene`).

Navigation and the arm planner sit low in the stack, but agents call them directly, so their tests
are agent tests. Their internals (the Nav2 costmap, the planner's robot model) are read on the
test side.

## The ladder

| Rung | Test (suite name) | Shape | What the robot does | Needs |
| --- | --- | --- | --- | --- |
| 1 | `scene_reset` | component | Spawns, resets several times, takes and loses ownership, e-stops (`check_world_lifecycle.py`) | |
| 2 | `sensors` | component | Stands facing a known box while an obstacle is thrown past; RGB-D alignment, frame rate, permissions (`check_observations.py`) | |
| 3 | `locomotion` | component | Walks, strafes, turns, stops, walks into a barrier (`check_locomotion.py`, `--manual`) | |
| 4 | `footprint` | component | Walks and turns while its outline is measured (`check_footprint.py`) | |
| 5 | `arm` | component | Raises, sweeps, opens and closes the gripper, cancels, resets (`check_manipulation.py`, `--manual`) | |
| 6 | `navigation_course`, `navigation_house` | agent | Walks to goals: around a wall, to a heading, by velocity commands; goals in a wall, in a closed pen or in furniture are refused; a walk is cancelled (`check_navigation.py`, specs `nav_course`, `nav_house`) | ROS 2 |
| 7 | `obstacle_avoidance` | agent | A box is moved into its route on the camera map; a box put in view, then taken away, must appear and clear (`check_dynamic_obstacle.py`, spec `nav_dynamic`) | ROS 2 |
| 8 | `arm_planning_known` (`_moveit`) | agent | Looks at the target, moves the hand over the clutter and a free spot, cancels, stows, walks; plans on the known boxes (`check_arm_planning.py`, spec `table_reach`) | cuRobo or MoveIt |
| 9 | `arm_planning_depth` (`_moveit`) | agent | The same with plans on depth from the cameras (spec `table_reach_depth`) | cuRobo or MoveIt |
| 10 | `pick_place_known`, `pick_place_depth` (`_moveit`) | agent | Picks the target next to the clutter, carries it over the clutter, places it on a free spot; on known boxes a carry only the held object would hit is refused (`check_pick_place.py`) | cuRobo or MoveIt |
| 11 | `drawer_grid`, `drawer_house` | agent | Opens the target's drawer by its handle and pushes it shut (`check_drawer_stroke.py`) | ROS 2, cuRobo |
| 12 | `drawer_place` | agent | Carries the target from the table into a drawer and closes it (`check_drawer_place.py`) | ROS 2, cuRobo |
| 13 | `search_house` | component | Explores the house kitchen until a colour detector sees the cube (`check_nav2_pipeline.py`, which runs `check_house.py`) | explore_lite overlay |
| 14 | `retrieve_grid`, `retrieve_house` | agent | Finds the target in a closed drawer, opens it, picks the target, stows (`check_retrieve.py`) | ROS 2, cuRobo |
| 15 | `entry_point` | agent | The template agent through `examples/run_world.py` on a skills-tier and a joints-tier spec (`check_entry_point.py`) | ROS 2, cuRobo |

Every rung takes `--robot`. A new robot starts before rung 1 with `components/check_bring_up.py`
(spawn, drive, arm, gripper, cameras; every value from its `robot.yaml`), then rung 4 for its
footprint ([adding a robot](../docs/adding_a_robot.md)).

Rung 13 stays a component test until the episode harness has a search goal: specs know only
`retrieve` and `none`, and an agent has no call to declare a found target.

## Running

```bash
source /opt/explore_ws/install/setup.bash  # ROS 2 Jazzy plus explore_lite, for rungs 6, 7, 11-15.
# The house rungs need its release fetched once: phygs.scene.fetch fetch --scene_id bear_loop_house --scene_version 1.1.0

# The ladder up to a rung, with videos; open <folder>/index.html for results and videos.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/run_suite.py ladder --upto 10 --record

# A group (base, nav2, manipulation, episodes) or single tests by suite name.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/run_suite.py manipulation --only pick_place_known

# One test directly.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_pick_place.py \
  --spec scripts/PhyGS/tests/robot_tests/configs/episodes/table_reach.yaml \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_pick_place --record
```

The suite writes `output_check/<YYYYMMDD-HHMM>_<label>/` with one folder and log per test,
`summary.json` (status and every measured value) and `index.html`. A test whose requirement is
missing is skipped, not failed. The last line names the first failing rung.

Every test takes `--output DIR` (a fresh folder), runs headless by default, writes
`DIR/check.json` (`test`, `status`, `error`, `checks` and the test's data) and exits 0 only when
it passed. Agent tests also leave the episode's `report.json` and `audit.jsonl`.

## Adding a rung

1. Agent test: a spec in `configs/episodes/`, a script in `episodes/` with its plan (session
   only, steps named with `episode.Steps`) and its checks, thresholds in `configs/<name>.yaml`
   with where each value came from. Component test: a script in `components/` that builds the
   robot with `common.robot_scene()`.
2. An entry in `LADDER` in `run_suite.py` with its rung and requirements.
3. A row in the table above.
