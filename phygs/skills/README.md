# Skills Package

Agents use the public APIs in this package. One folder per client: `<name>/api.py` is the thin
client, `<name>/execution/` the machinery behind it, and `RobotControl` (`phygs/world/robot.py`)
binds the clients to the runtime.
Layers, the runtime tick and data flow are in [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Package Layout

| Path | Responsibility | Status |
| --- | --- | --- |
| `locomotion/api.py` | Body velocity (all zeros stops) and feedback; `execution/`: the base controllers. | Live |
| `navigation/api.py` | `navigate_to`, `explore`, cancellation and feedback; `execution/`: Nav2, the maps, the footprint, `launch/` (Nav2's launch file and parameters). | Live |
| `manipulation/api.py` | Joint, pose (`move_to_pose`), plan, joint and end-effector trajectory, gripper and cancel commands; the `stow`, `unstow` and `look_at` skills; `execution/`: the arm controller, the skill backends, the planners. | Live |

## API Boundary

Agents receive an assembled robot and call its clients. The World
owns `robot.step()`:

```python
command = robot.navigation.navigate_to(x=3.8, y=0.0, heading=0.0)
feedback = robot.navigation.feedback(command)
robot.locomotion.command_velocity(0.4, 0.0, 0.0, duration=0.5)
command = robot.manipulation.move_joints({"arm_sh0": 0.3}, duration=3.0)
observations = robot.observe()
```

Long-running commands return command ids; `feedback(id)` reports their status
(`active`, `stopping`, `succeeded`, `failed`, `cancelled`) without blocking hidden loops.
The Spot-SDK-shaped facade and the AO-Grasp/CuRobo runner were removed on 2026-10-08.
The older Lula/RMPflow pick-and-place demos in `manipulation/` were removed on 2026-10-16.

## Adding a Skill

Follow the steps in [ARCHITECTURE.md](../../ARCHITECTURE.md#extension-points).

## Rules

- Fail fast on invalid robot state, missing target data, missing planner output, or missing sensor data.
- Keep public APIs stable for benchmark scripts.
- Avoid broad `try/except` blocks that convert real failures into default behavior.
- Do not make vendor-specific APIs leak into benchmark launchers unless wrapped by a skill or service client.
