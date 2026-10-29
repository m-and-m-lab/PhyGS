# The world: what a benchmark drives

A benchmark says what is asked and how it is scored; PhyGS runs the robot in the world and measures
what happened. This page is the benchmark author's: build a world from a spec, pass a task, run an
agent, read the report. Everything here is `phygs.world` (and `phygs.scene` for scenes by id).

## One run, in code

```python
from phygs.world import EpisodeSpec, World, load_agent   # World after the Isaac Sim app has started

spec = EpisodeSpec.load("episode.yaml")                  # validated before Isaac Sim starts; scene by id
task = my_benchmark.load_task(spec.task)                 # brief, privileged target, result(), provenance()
name, agent = load_agent("template")                     # an example agent by name, or any run(session) function
with World(spec, output_folder, task=task, clock="charged",
           overrides={"end_settle_seconds": 1.0, "planner": {"backend": "curobo"}}) as world:
    report = world.run(agent, name=name)                 # stops the robot, keeps the grip, settles, evaluates
```

`examples/run_world.py` is this with argument parsing; `phygs run` is its command. `phygs.world`
imports each name on first use, so `EpisodeSpec`, `Task` and the report schema work before the app
starts and only `World` (with `Truth` and `run_episode`) needs it.

## The pieces

| Piece | Module | What it is |
| --- | --- | --- |
| Spec | `world/spec.py` (`EpisodeSpec`, `SCHEMA = 3`, `load_agent`) | Scene by id and version, robot and start, objects, an opaque `task` block, actions, observations, grants, budget, seed. A benchmark adds its own blocks beside these and strips them before `from_dict`. |
| Task | `world/task.py` (`Task` protocol, `NoTask`, `load_task(block, loader)`) | `goal`, `description` (public), `target` (privileged), `brief()`, `result(truth, ended_by, contacts, drawers_open_at_start)`, `provenance()`. The framework never imports a task; it takes the object. |
| World | `world/world.py` (`World`, `run_episode`) | Builds the scene, robot, runtime, backends; runs the agent in a thread; audits; stops; evaluates; writes the report. `overrides` pins settings over `configs/environment.yaml` (`world/build.py`, `SETTINGS`). |
| Ground truth | `World.truth()` → `Truth` (`world/truth.py`) | Object poses and boxes, `held`, drawer openings, base pose, arm joints, stow error. Only the task's evaluator and robot tests read it. |
| Report | `world/report.py` (`REPORT_SCHEMA`, `KEYS`, `compose`), `world/recorder.py`, `world/provenance.py` | `report.json` keys with their meaning; `audit.jsonl`; the video; hashes of every asset and threshold and the code version. |
| Scenes | `scene/registry.py` (`scene_file(id, version)`), `scene/fetch.py` (`phygs scenes`) | Roots: `$PHYGS_SCENES`, `~/.cache/phygs/scenes`, the fixtures. [scenes.md](scenes.md). |
| Session | `phygs.session` (`AgentSession`, target types) | What the agent gets; a benchmark's agents import only this. [design](design/episode_harness/session.html). |

## Settings a benchmark owns

`configs/environment.yaml` names them in its header; a benchmark passes them as `overrides` and
records them in its own rules file, never a copy of the settings:

| Key | What it decides |
| --- | --- |
| `end_settle_seconds` | How long the robot settles after the stop before the evaluator reads ground truth [s]. |
| `stop_all_timeout_seconds` | How long the stop at the end may take [s]. |
| `contact_record_n` | Arm contacts above this force go into the report [N]. |
| `skills.backend` | `collision_avoidance_planner` or `direct`. |
| `planner.backend` | `curobo` or `moveit`. |

## What a benchmark owns, and what it does not

- Owns: the task object and its thresholds, the episode files and their extra blocks, which grants and actions an agent gets (a track), the settings above, how results are aggregated.
- Does not own: the robot, the physics rate, the cameras, the skills, the runtime, the report's keys, how a scene is read. A benchmark that needs one of those changed asks PhyGS for a release.

## Versions

A benchmark pins a PhyGS version (`phygs.__version__`, a tag). That version promises the spec schema
(`SCHEMA`) and the report schema (`REPORT_SCHEMA`); every report records both under `schemas`. A
change to either is at least a minor PhyGS release and a benchmark version change. 0.4.0 renamed the
modules a benchmark imports ([layout.md](layout.md#renamed-in-040) has the old and new paths).
