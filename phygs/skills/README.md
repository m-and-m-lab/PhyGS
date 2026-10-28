# Skills Package

Agents use the public APIs in this package. Each `<name>/api.py` is a thin client;
execution lives under `phygs/execution`, and `RobotControl`
(`phygs/robots/control.py`) binds the clients to the runtime.
Layers, the runtime tick and data flow are in [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Package Layout

| Path | Responsibility | Status |
| --- | --- | --- |
| `locomotion/api.py` | Public velocity, stopping and feedback interface. | Live |
| `navigation/api.py` | Public navigation, cancellation and feedback interface. | Live |
| `manipulation/api.py` | Public joint, pose (`move_to_pose`), joint and end-effector trajectory, gripper and cancel commands; `stow` and `look_at` skills. | Live |

## API Boundary

Agents receive an assembled robot and call its clients. The environment runner
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
