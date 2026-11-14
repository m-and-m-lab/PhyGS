# robots

One folder per robot, and the description that loads it.

| Path | Holds |
| --- | --- |
| `description.py` | `RobotDescription`: every robot fact the framework reads, loaded and checked from a folder's `robot.yaml` (no Isaac imports); `RobotInfo`, what agents read as `session.robot`. |
| `spot/` | Spot with arm: `robot.yaml`, the leg and arm actuators, the walking policy wrapper, the policy, the transit footprint, the assets. See [spot/README.md](spot/README.md). |
| `ridgeback_franka/` | The second robot: Ridgeback base with a Franka arm. See [ridgeback_franka/README.md](ridgeback_franka/README.md). |
| `_template/robot.yaml` | A commented starting point for a new robot. |

The framework names no robot's parts; a robot's own code is only its base controller and its
actuators. To add a robot: [docs/adding_a_robot.md](../../docs/adding_a_robot.md). The robot is
assembled with its skill clients in `phygs/world/robot.py` (`RobotControl`).

Up: [phygs/README.md](../README.md). Tests: `tests/test_robot_description.py`,
`tests/test_robot_neutral.py`, `tests/test_second_robot.py`, `tests/test_spot_actuators.py`; the
robot tests' bring-up check is `tests/robot_tests/components/check_bring_up.py`.
