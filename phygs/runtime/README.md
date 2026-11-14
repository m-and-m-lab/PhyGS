# runtime

The one place that owns time and joints in an episode.

| File | Holds |
| --- | --- |
| `episode.py` | `EpisodeRuntime`: the integer clock, controller declarations, exclusive joint ownership, human authority, reset, emergency stop, events; `JointTargets`, `EpisodeStatus`. |
| `commands.py` | Command bookkeeping: ids and episode-scoped feedback. |
| `timing.py` | Exact conversion between command durations and clock ticks. |
| `pipeline.py` | `ObservationPipeline`: capture, task-wide permissions, agent and service views, processors that reset and update with the episode. |

Every joint target passes through `EpisodeRuntime`. The World steps it once per physics tick; agents
never step physics. The tick, ownership and reset in detail:
[ARCHITECTURE.md](../../ARCHITECTURE.md#runtime-tick-ownership-and-reset). The pipeline's
arrangements: [docs/pipelines.md](../../docs/pipelines.md).

Up: [phygs/README.md](../README.md). Tests: `tests/test_episode_runtime.py`, `tests/test_pipeline.py`;
the lifecycle in the simulator: `tests/robot_tests/components/check_world_lifecycle.py`.
