# world

What a benchmark drives: a `World` built from an `EpisodeSpec`, a `Task` it is given, an agent it
runs, and the report it writes. Also where the robot is assembled.

| File | Holds |
| --- | --- |
| `world.py` | `World` (build, reset, run, stop, evaluate, report) and `run_episode`. Needs Isaac Sim. |
| `spec.py` | `EpisodeSpec` (`SCHEMA`) and `load_agent`; no Isaac imports. |
| `task.py` | The task seam: the `Task` protocol, `EpisodeEnd`, `NoTask`, `load_task`. |
| `truth.py` | `Truth`: ground truth for the task's evaluator and the robot tests, never for agents. |
| `robot.py` | `RobotControl`: the robot's controllers, runtime, pipeline and skill clients, assembled once. |
| `build.py`, `environment.yaml` | The pure build steps, and every setting (the benchmark-owned ones named in its header). |
| `start.py` | The check before the agent starts. |
| `navigation.py` | Nav2 for the World: the footprint check, the map source, the servers. |
| `report.py`, `recorder.py`, `provenance.py` | The report's keys and schema; the run's output folder; hashes and the code version. |
| `viewer.py`, `replay.py` | A read-only view of a running episode (the robot UI uses it); rendering a finished run again. |

`phygs.world` imports each name on first use, so `EpisodeSpec`, `Task` and the report schema work
before the Isaac Sim app starts; only `World`, `Truth` and `run_episode` need it. The benchmark
author's guide: [docs/world.md](../../docs/world.md). One run, step by step:
[PIPELINE.md](../../PIPELINE.md). Entry point: `phygs run`
([examples/run_world.py](../../examples/run_world.py)).

Up: [phygs/README.md](../README.md). Tests: `tests/test_episode_spec.py`, `tests/test_world_build.py`,
`tests/test_start_check.py`, `tests/test_episode_recorder.py`, `tests/test_episode_provenance.py`;
in the simulator, the agent tests in `tests/robot_tests/episodes/`.
