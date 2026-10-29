# The world: what a benchmark drives

A benchmark says what is asked and how it is scored; PhyGS runs the robot in the world and measures
what happened. This page is the benchmark author's: build a world from a spec, pass a task, run an
agent, read the report. Everything here is under `phygs.episodes` today (the layout proposal renames
it `phygs.world` and `phygs.session`; the names below keep working either way).

## One run, in code

```python
from phygs.episodes.environment import Environment      # after the Isaac Sim app has started
from phygs.episodes.spec import EpisodeSpec, load_agent
from phygs.episodes.task import NoTask, Task             # the seam: what the Environment needs from a task

spec = EpisodeSpec.load("episode.yaml")                  # validated before Isaac Sim starts; scene by id
task = my_benchmark.load_task(spec.task)                 # brief, privileged target, result(), provenance()
name, agent = load_agent("template")                     # or any run(session) function
with Environment(spec, output_folder, task=task, clock="charged",
                 overrides={"end_settle_seconds": 1.0, "planner": {"backend": "curobo"}}) as world:
    report = world.run(agent, name=name)                 # stops the robot, keeps the grip, settles, evaluates
```

`examples/run_world.py` is this with argument parsing; `phygs run` is its command.

## The pieces

| Piece | Module | What it is |
| --- | --- | --- |
| Spec | `episodes/spec.py` (`EpisodeSpec`, `SCHEMA = 3`) | Scene by id and version, robot and start, objects, an opaque `task` block, actions, observations, grants, budget, seed. A benchmark adds its own blocks beside these and strips them before `from_dict`. |
| Task | `episodes/task.py` (`Task` protocol, `NoTask`, `load_task(block, loader)`) | `goal`, `description` (public), `target` (privileged), `brief()`, `result(truth, ended_by, contacts, drawers_open_at_start)`, `provenance()`. The framework never imports a task; it takes the object. |
| Scenes | `episodes/scenes.py` (`scene_file(id, version)`), `scenes/fetch.py` | Roots: `$PHYGS_SCENES`, `~/.cache/phygs/scenes`, the fixtures. [scenes.md](scenes.md). |
| Environment | `episodes/environment.py` (`Environment`, `run_episode`) | Builds the scene, robot, runtime, backends; runs the agent in a thread; audits; stops; evaluates; writes the report. `overrides` pins settings. |
| Ground truth | `Environment.truth()` → `Truth` | Object poses and boxes, `held`, drawer openings, base pose, arm joints, stow error. Only the task's evaluator and robot tests read it. |
| Session | `episodes/session.py` (`AgentSession`, `SESSION_SCHEMA`) | What the agent gets. [design](design/episode_harness/session.html). |
| Report | `episodes/report.py` (`REPORT_SCHEMA`, `KEYS`), `episodes/recorder.py`, `episodes/provenance.py` | `report.json` keys with their meaning; `audit.jsonl`; the video; hashes of every asset and threshold and the code version. |

## What a benchmark owns, and what it does not

- Owns: the task object and its thresholds, the episode files and their extra blocks, which grants and actions an agent gets (a track), the end-of-episode settings and backend pins (`overrides`), how results are aggregated.
- Does not own: the robot, the physics rate, the cameras, the skills, the runtime, the report's keys, how a scene is read. A benchmark that needs one of those changed asks PhyGS for a release.

## Versions

A benchmark pins a PhyGS version (`phygs.__version__`, a tag). That version promises `SCHEMA` (spec), `SESSION_SCHEMA` and `REPORT_SCHEMA`; the report records them. A change to any of them is at least a minor PhyGS release and a benchmark version change.
