# Adding a robot

How to run the framework with another mobile manipulator: one folder that describes the robot, a
plugin only where its physics differs, and a short series of checks to bring it up. Design and
reasons: [robot_portability/index.html](robot_portability/index.html).

## What a robot is

A robot is one folder with a `robot.yaml`. The framework reads every robot fact from it: the asset,
the joints, the gripper and its tool frames, the cameras and their roles, the limits and the
measured calibrations. Framework code never names a robot's parts; it asks the description
(`phygs/robots/description.py`, `RobotDescription`). A unit test fails when Spot's
joint, link or camera names appear in framework code (`tests/test_robot_neutral.py`).

The folder can be anywhere. Robots of this repository are in `phygs/robots/<name>/`:

| Robot | Base | Status |
| --- | --- | --- |
| [spot](../phygs/robots/spot/README.md) | Legs, learned walking policy | Reference robot: every check |
| [ridgeback_franka](../phygs/robots/ridgeback_franka/README.md) | Planar joints (omnidirectional) | Bring-up, footprint and navigation pass; the arm tasks are next |

Pick a robot by name or by path. An episode spec names its robot (`robot.model`); `--robot` runs
the same episode, test or UI with another one:

```bash
./isaaclab.sh -p scripts/PhyGS/examples/run_world.py \
  --spec scripts/PhyGS/tests/robot_tests/configs/episodes/dev_grid_table.yaml \
  --agent template --robot /path/to/my_robot/robot.yaml \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_my_robot
```

## Steps

Run the commands from `/workspace/isaaclab`. `R` is your robot's name (a folder in
`phygs/robots/`) or the path of its `robot.yaml`.

1. **Copy the template.** Every field of `robot.yaml` has a comment.

   ```bash
   cp -r scripts/PhyGS/phygs/robots/_template \
     scripts/PhyGS/phygs/robots/my_robot
   ```

2. **Fill `robot.yaml`.** Every field is required: a missing field is an error, never Spot's value
   (in `arm.motion` only the limits and tolerances in joint units are; the Cartesian tolerances and
   the waits may keep the controller's defaults). Name the arm joints in chain order, the gripper
   joints with the primary first, the [tool frames](#tool-frames) and the
   [camera roles](#camera-roles). Leave `navigation.footprint_profile` null until step 6.

3. **Give the planners a URDF** rooted at the base link, with collision meshes.
   `ridgeback_franka/make_planning_urdf.py` writes one from the USD's joint frames.

4. **Run the description check** until it passes. No simulator: it checks `robot.yaml` against
   the URDF and the USD (joints, links, chain order, limits, camera prims, tool frames).

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest \
     scripts/PhyGS/tests/test_robot_description.py -q -p no:cacheprovider
   ```

   For a robot outside the repository: `load_robot(path).validate_urdf()` and `.validate_usd()`.

5. **Bring it up in the simulator.** It spawns with the arm stowed, drives forward, sideways and
   turns (at 60 % of `navigation.limits`), stops, moves the arm to the pre-stow, closes and opens the
   gripper, reads every camera and checks that localization follows the base link.

   ```bash
   ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_bring_up.py \
     --robot R --record --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_bring_up
   ```

6. **Measure the footprint.** Navigation refuses to start without it. The check drives the robot
   with the arm stowed and writes `transit.json`; copy it into the robot's folder and set
   `navigation.footprint_profile` to it.

   ```bash
   OUT=scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_footprint
   ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_footprint.py \
     --robot R --record --output $OUT
   ROBOT=scripts/PhyGS/phygs/robots/my_robot
   mkdir -p $ROBOT/navigation && cp $OUT/transit.json $ROBOT/navigation/
   # robot.yaml: navigation.footprint_profile: navigation/transit.json
   ```

7. **Navigate.** The course: around a wall, to a heading, by velocity commands, goals refused in a
   wall and in a closed pen, a cancel. Needs ROS 2 Jazzy sourced (`source /opt/ros/jazzy/setup.bash`).

   ```bash
   ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_navigation.py \
     --spec scripts/PhyGS/tests/robot_tests/configs/episodes/nav_course.yaml \
     --robot R --record --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_navigation
   ```

8. **Arm tasks.** The rest of the robot-test ladder takes `--robot` too
   ([tests/robot_tests/README.md](../tests/robot_tests/README.md); `run_suite.py ladder`): arm
   planning, pick and place, drawers, retrieve. Their scripted routines place your
   [tool frames](#tool-frames); where the robot stands in a scene (`configs/stances/<scene>.yaml`)
   and the routines' strategy (`routines/params.yaml`) were tuned for Spot.

9. **Write down what passed**, with dates and output folders, in the robot folder's `README.md`
   (see `ridgeback_franka/README.md`).

## Tool frames

`gripper.tools` names frames on the gripper, as poses in the wrist link (`position` [m],
`rotation_xyzw`). The skills (`move_to_pose(tool=...)`, `tool_pose`), the UI's Move to pose and the
robot tests' routines place these frames, so no code assumes how a gripper is mounted.

`wrist` is the wrist link itself. Every other tool has the **gripper's axes**, one rotation for all
of them (the description check enforces it):

- **x**: the approach, out of the palm toward the object;
- **z**: the axis the jaws close along;
- **y**: z × x.

| Tool | Required | What goes there |
| --- | --- | --- |
| `hand` | yes | The middle of the jaw gap: the default of `move_to_pose(tool=)`. |
| `grasp` | yes | An object's grasp point: the routines' picks and the UI's Move to pose put the clicked or chosen point here. |
| `handle` | for drawer routines | A bar handle in a pull grasp: its axis, or, with a fixed jaw, its underside. |
| `fixed_jaw` | if the gripper has one | The fixed jaw's inner pad: a held object rests on it. Without it the jaws centre the object. |

The same convention, two grippers:

```yaml
# Spot: the gripper's axes are the wrist link's (x the approach, the claw closes along z).
tools:
  wrist: {position: [0.0, 0.0, 0.0]}
  hand: {position: [0.20, 0.0, 0.029]}
  grasp: {position: [0.183, 0.0, 0.029]}
  handle: {position: [0.22, 0.0, -0.029]}
  fixed_jaw: {position: [0.20, 0.0, -0.036]}

# Franka hand: it approaches along its z and closes the fingers along its y, so the tools turn
# the hand's axes (x <- z, y <- x, z <- y).
tools:
  wrist: {position: [0.0, 0.0, 0.0]}
  hand: {position: [0.0, 0.0, 0.1034], rotation_xyzw: [-0.5, -0.5, -0.5, 0.5]}
  grasp: {position: [0.0, 0.0, 0.1034], rotation_xyzw: [-0.5, -0.5, -0.5, 0.5]}
  handle: {position: [0.0, 0.0, 0.1034], rotation_xyzw: [-0.5, -0.5, -0.5, 0.5]}
```

## Plugin points

Two places take code. Everything else is data.

| Point | `robot.yaml` | Contract |
| --- | --- | --- |
| Base controller | `base.kind` | `legged_policy` (a walking policy; `base.policy` names its wrapper, `base.config` its settings: Spot), `planar_joints` (velocity-driven x, y, yaw joints: Ridgeback), or your class as `module:Class` / `file.py:Class`. A class implements `BaseController` (`execution/locomotion/base.py`) and builds itself with `from_robot(robot, articulation, *, physics_dt, source)`; `files(robot)` names the files its motion depends on (they are hashed into the footprint profile and the report). |
| Actuator models | `actuators` | A function `(physics_dt) -> {name: Isaac Lab actuator config}`, as `file.py:function` in the folder or `package.module:function`. |

A `file.py:attribute` reference loads that file of the robot's folder, so a robot outside the
repository needs no package.

## Camera roles

| Role | Used by |
| --- | --- |
| `navigation` | Nav2's voxel layer maps their point clouds. |
| `planning` | The arm planner's depth map without `known_map` (one image size for all). |
| `look` | The camera `look_at` aims. |
| `front` | The UI's and the video's Front view: one camera, or a left and a right one to stitch. |

Agents see the robot through `session.robot` (`RobotInfo`): arm and gripper joints, tool frames,
cameras, the look camera and the stow.

## What does not work yet

- A differential base: `command_velocity` refuses `v_y`, and navigation refuses to start (the fine
  approach moves sideways; Nav2's DiffDrive model is ready in the launcher).
- More than one arm or gripper, and whole-body controllers that move the base for the arm.
- The routines' stances per scene and their strategy were tuned for Spot; another robot may need
  its own stances to reach the drawers and tables.
