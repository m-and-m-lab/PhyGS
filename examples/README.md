# Examples

Scripts that run PhyGS, and agents to copy. Up: [repository README](../README.md). Agent contract:
[docs/agents.md](../docs/agents.md).

| Path | What it does | Command |
| --- | --- | --- |
| `run_world.py` | Runs one episode: builds the world from a spec, runs an agent, writes the report and the recording. | `phygs run` |
| `replay_run.py` | Renders a finished run again from its recording (`episode.rrd`) into `<run>/replay`. | `phygs replay <run>` |
| `replay_plans.py` | Plans a finished run's arm plans again, offline, from its recording, to compare planner versions or settings. | `isaaclab -p scripts/PhyGS/examples/replay_plans.py <runs>` (options in the file's docstring) |
| `agents/template.py` | A starting agent: one call on every client the spec grants, then finish. Copy it and edit `run(session)`. | `phygs run --agent template` |
| `agents/depth_example.py` | An agent that walks to a drawer, looks at its handle with the hand camera and locates the handle from depth. | `phygs run --agent depth_example` |

An agent is a `run(session)` function that imports only `phygs.session`
(`tests/test_agent_imports.py` checks the examples). `--agent` takes a name from `agents/`, or
`file.py:function`.

The `phygs` command is defined in [phygs/__main__.py](../phygs/__main__.py); most subcommands run
the script they name with the remaining arguments, so `phygs run --help` is `run_world.py --help`
(`scenes` runs the `phygs.scene.fetch` module; `version` prints the version).

Tests: `tests/test_agent_imports.py` (the examples import only the session) and
`tests/test_episode_spec.py` (the template loads by name). The example specs are the robot tests'
([tests/robot_tests/configs/episodes/](../tests/robot_tests/configs/episodes/)).
