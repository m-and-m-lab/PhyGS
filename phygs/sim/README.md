# sim

Adapters to Isaac Lab. Everything here needs a running Isaac Sim app.

| File | Holds |
| --- | --- |
| `state.py` | `IsaacLabWorld`: state restoration, and the one actuator-write and physics-step boundary. |
| `robot_scene.py` | The robot's articulation, cameras and IMU from its description, the ground and the light; the simulation settings. |
| `observations.py`, `cameras.py` | Measurement capture and publication rates; cameras rendered by RTX and read from Replicator annotators. |
| `base.py`, `arm.py`, `footprint.py` | The base link's measured pose and velocity; differential IK for the wrist link; live collision bounds. |
| `fabric.py` | Fabric pose writes, kept only for the steps that rendering reads. |
| `gpu.py` | The GPU Isaac Sim renders on, shared by the cameras' readers and the arm planner's worker. |

Simulator objects never reach an agent: the adapters publish plain measurements through the
observation pipeline. Where this sits: [ARCHITECTURE.md](../../ARCHITECTURE.md#at-a-glance);
what is simulated: [docs/physics.md](../../docs/physics.md).

Up: [phygs/README.md](../README.md). Tests: `tests/test_footprint.py` and the fakes in
`tests/fakes.py` for the pure parts; the simulator itself is checked by the robot tests,
`tests/robot_tests/components/` (`check_world_lifecycle.py`, `check_observations.py`,
`check_footprint.py`).
