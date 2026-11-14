# recording

How a run is written down and shown again. Nothing here acts on the robot.

| File | Holds |
| --- | --- |
| `log.py` | `EpisodeLog`: writes the run's recording, `episode.rrd` (a [Rerun](https://rerun.io) file), through the Rerun SDK. |
| `reader.py` | `Recording`: reads an `episode.rrd` back (properties, agent calls, states, camera images, arm plans). |
| `blueprint.py` | The viewer's layout for a recording. |
| `video.py`, `panels.py`, `overview.py` | The run's video: a camera following the robot, the panels beside it and the overview strip under it. |
| `timeline.py`, `narration.py` | Commands as spans on a base and an arm lane; the skills' step lines. |
| `trace.py` | A run's timeline and path (`trace.json`), read from its output folder. |

Entry points: `phygs replay <run>` ([examples/replay_run.py](../../examples/replay_run.py)) renders a
run from its recording; `rerun episode.rrd` opens it. What the recording holds, and the replay's
outputs: [docs/world.md](../../docs/world.md#what-a-run-leaves-and-replaying-it).

Up: [phygs/README.md](../README.md). Tests: `tests/test_episode_log.py`,
`tests/test_recording_reader.py`, `tests/test_recording_panels.py`, `tests/test_video_timeline.py`,
`tests/test_narration.py`, `tests/test_trace.py`.
