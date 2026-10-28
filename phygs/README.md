# API and execution layout

```text
phygs/
├── agents/                  # Example agents: run(session); they import only episodes.session
├── configs/                 # environment.yaml (physics, cameras, Nav2, planner, timing); navigation/, ui/, robots/, scenes/
├── episodes/                # Specs (scene by id), Environment, AgentSession, hints, the Task seam, recorder, provenance
├── execution/
│   ├── locomotion/          # BaseController contract; legged_policy and planar_joints base controllers
│   ├── navigation/          # Nav2 backend, footprint, launch/nav2.launch.py, shared types and geometry
│   └── manipulation/        # Arm runtime, end-effector trajectories, skills, cuRobo and MoveIt planners
├── observations/            # Observation packets, transforms and localization
├── robots/                  # One folder per robot; descriptions, scene, assembly
│   ├── description.py       # RobotDescription: robot.yaml, loaded and checked
│   ├── scene.py             # The robot's Isaac Lab scene and simulation settings
│   ├── control.py           # RobotControl: base controller, runtime, clients
│   ├── _template/robot.yaml # Every field, commented: start a new robot here
│   ├── spot/                # robot.yaml, actuators, walking policy, calibrations, assets/ (USD, URDF, spot_description)
│   └── ridgeback_franka/    # robot.yaml, planning URDF, asset fetch (in bring-up)
├── runtime/                 # Clock, ownership, expiry, reset and termination
├── scenes/                  # fetch.py: ForageBench scene releases into the local scene cache
├── simulation/              # Isaac Lab physical state, sensors, USD scenes and map adapters
├── skills/                  # Public clients: locomotion, navigation, manipulation (api.py each)
├── tools/                   # Video, panels, timeline, narration, colour detection, collision map
└── ui/                      # The robot UI (a person as the agent): server, bridge, session, cli.py
```

Agents import acting interfaces from `phygs.skills.locomotion.api`,
`phygs.skills.navigation.api`, and `phygs.skills.manipulation.api`; the assembled robot exposes those clients:
```python
command = robot.navigation.navigate_to(x=3.8, y=0.0, heading=0.0)
feedback = robot.navigation.feedback(command)
robot.navigation.cancel(command)
robot.locomotion.command_velocity(0.4, 0.0, 0.0, duration=0.5)
robot.locomotion.stop()
observations = robot.observe()
```

`api.py` submits locomotion commands through the shared runtime. The internal
controller computes leg targets. Navigation delegates planning and following to
`phygs.execution`. Agents do not import those internals. Shared navigation types live under
execution and are exported by the public API so agents need only one import.

A robot is a folder with a `robot.yaml` (`robots/description.py` loads and checks it;
[adding a robot](../docs/adding_a_robot.md)). `robots/scene.py` spawns it with its sensors;
`robots/control.py` assembles its base controller, the runtime, execution components and public
skill clients. The environment runner
owns reset, stepping and close. Existing diagnostic arm control remains here until
manipulation is migrated.

`setup.py` packages both source roots into one distribution. Repository diagnostics
add `scripts` and `src` to their import path; installed callers need no path edits.
Basic manipulation execution lives in `execution/manipulation/controller.py`,
with its Isaac Lab wrist adapter in `simulation/arm.py`. The older SDK/manipulation
experiments and their helpers remain isolated under
`scripts/skills` and `scripts/helpers`. Importing the locomotion/navigation APIs
does not import that stack or ROS.

Robot-specific code and data (Spot's policy, actuators and calibrations) live in the robot's
folder; framework code names no robot's parts (`tests/test_robot_neutral.py`). Runnable physical checks are `phygs_robot_tests/episodes/check_*.py` and
`phygs_robot_tests/components/check_*.py`, automatic and headless by default (`--manual` for
keyboard control where a test has it); component tests read `phygs_robot_tests/configs/`, and the
robot is `--robot` (default `spot`). The first object-search evaluator lives under
`src/phygs_robot_tests/tasks/object_search.py` and its agent-facing declaration
client lives under `phygs_robot_tests/tasks/object_search_api.py`.

Optional input/processing arrangements and navigation factory migration are documented
in [pipelines](../docs/pipelines.md).
