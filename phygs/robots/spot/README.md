# Spot

Boston Dynamics Spot with the arm and six Orbbec Gemini 2 depth cameras: the robot the framework was
built on. Everything the framework reads about it is in `robot.yaml`; the rest of this folder is its
own code and data.

| File | What |
| --- | --- |
| `robot.yaml` | The description: asset, joints, gripper, tools, cameras and their roles, limits. |
| `actuators.py` | The actuator groups: the twelve legs as one fused group (`SpotLegActuator`) and the arm drives. |
| `policy.py` | The walking policy wrapper (`SpotFlatTerrainPolicy`): observations, stand switch, gains. |
| `locomotion/` | The policy (`policy.pt`, its training `env.yaml`) and `locomotion.yaml`: stand drive, stand switch, stop, dead band, deceleration and lowest-speed moves, each with the run it was measured in. |
| `navigation/transit.json` | The measured transit footprint (`check_footprint.py`, 2026-10-22); its hashes pin the USD, the policy and its configuration. |

The USD, URDF and cuRobo configuration are in `assets/` (the `spot_description` submodule beside them).

## Bring-up ladder

Spot is the reference robot of the robot-test ladder (`phygs_robot_tests/run_suite.py`): all 27
suite tests passed on 2026-11-09 at commit 33543b3 (each test's last evidence is in
`tests/robot_tests/ladder.json`). After a change to the asset or the policy, re-measure
`navigation/transit.json` (rung 03); navigation refuses a profile whose hashes do not match.
