# The world: what a benchmark drives

A benchmark says what is asked and how it is scored; PhyGS runs the robot in the world and measures
what happened. This page is the benchmark author's: build a world from a spec, pass a task, run an
agent, read the report. Everything here is `phygs.world` (and `phygs.scene` for scenes by id).

## One run, in code

```python
from phygs.world import EpisodeSpec, World, load_agent   # World after the Isaac Sim app has started

spec = EpisodeSpec.load("episode.yaml")                  # validated before Isaac Sim starts; scene by id
task = my_benchmark.load_task(spec.task)                 # brief, privileged target, reset/update/result, provenance
name, agent = load_agent("template")                     # an example agent by name, or any run(session) function
with World(spec, output_folder, task=task, clock="charged",
           overrides={"end_settle_seconds": 1.0, "planner": {"backend": "curobo"}}) as world:
    report = world.run(agent, name=name)                 # checks, resets the task, runs, stops, settles, evaluates
    world.next_episode(output_folder / "s1", seed=1)     # the same spec again, without building again
    report_1 = world.run(agent, name=name)
```

`examples/run_world.py` is this with argument parsing; `phygs run` is its command. `phygs.world`
imports each name on first use, so `EpisodeSpec`, `Task` and the report schema work before the app
starts and only `World` (with `Truth` and `run_episode`) needs it.

## The pieces

| Piece | Module | What it is |
| --- | --- | --- |
| Spec | `world/spec.py` (`EpisodeSpec`, `SCHEMA = 3`, `load_agent`) | Scene by id and version, robot and start, objects, an opaque `task` block, actions, observations, grants, budget, seed. A benchmark adds its own blocks beside these and strips them before `from_dict`. |
| Task | `world/task.py` (`Task` protocol, `NoTask`, `load_task(block, loader)`) | `goal`, `description` (public), `target` (privileged), `brief()`, `reset(truth)` (the settled world, before the agent starts), `update(truth)` (every `evaluation_period_seconds` while the agent runs), `result(truth, end)` (once, after the stop and settle; `end` is an `EpisodeEnd`: how it ended, the simulated time, the unintended arm contacts), `provenance()`. The framework never imports a task; it takes the object. |
| World | `world/world.py` (`World`, `run_episode`) | Builds the scene, robot, runtime, backends; checks the settled world before the agent starts (`world/start.py`); runs the agent in a thread; audits; stops; evaluates; writes the report. `overrides` pins settings over `configs/environment.yaml` (`world/build.py`, `SETTINGS`). |
| Ground truth | `World.truth()` → `Truth` (`world/truth.py`) | Object poses and boxes, `held`, `pinched` (every jaw pressing on a body), arm contacts, the drawer a body started in, how far bodies moved from where they were placed, drawer openings, base pose, arm joints, stow error. Only the task's evaluator and robot tests read it. |
| Report | `world/report.py` (`REPORT_SCHEMA`, `KEYS`, `compose`), `world/recorder.py`, `world/provenance.py` | `report.json` keys with their meaning; `audit.jsonl`; the video; hashes of every asset and threshold and the code version. `status` says whether the episode can be scored: `completed`, `setup_error` (the start check failed, the agent never ran) or `error` (the simulator or the World broke). Only `completed` episodes count in a rate. |
| Scenes | `scene/registry.py` (`scene_file(id, version)`), `scene/fetch.py` (`phygs scenes`) | Roots: `$PHYGS_SCENES`, `~/.cache/phygs/scenes`, the fixtures. [scenes.md](scenes.md). |
| Session | `phygs.session` (`AgentSession`, target types) | What the agent gets; a benchmark's agents import only this. |

## What a run leaves, and replaying it

A run's output folder holds:

| File | What it is |
| --- | --- |
| `report.json` | The report (`world/report.py`). |
| `audit.jsonl` | One line per agent call (time, call, arguments, result or error), and the agent's named steps: `with session.step("open the drawer"):` writes `session.step` lines where the step begins and ends (with the error that left it). Steps nest; they are not calls and are not counted in the report's `usage`. |
| `state.npz` | The state log (`recording/state_log.py`), on every run: every camera period, the robot's root pose and joints, whether the gripper holds something, the base's commanded velocity, and the pose of each scene body that moved since its last logged pose; Nav2's route and global costmap when they change (the costmap read at most once per simulated second); each arm plan's display (the recording's arm planner panel); the command timeline's spans. Arrays only, nothing rendered, so it costs the run no speed. |
| `third_person.mp4` | With `--record` only: the video, rendered while the run goes (it slows the run), in the same look as `phygs replay` (`recording/video.py`): a camera following the robot, the panels on its right, the overview strip under both with the timeline so far. |
| `nav2/`, `planner/` | The Nav2 servers' and the arm planner's logs. |

`phygs replay <run>` (`examples/replay_run.py`) renders the run afterwards from these files: it builds
the scene and robot again without control, Nav2 or the planner, poses them sample by sample without
stepping physics, and writes into `<run>/replay`: `replay.mp4`, laid out like the `--record` video (a
camera following the robot, with the recording's panels on its right: front and hand cameras, their
depth, navigation with Nav2's costmap and route, and the arm planner), with the overview strip under it
(clock, what the gripper holds, contact force, the agent's step, the floor plan with the path walked and
Nav2's route, and the timeline of steps, commands, grips and bumps); `trace.json` (the timeline and path
the strip is drawn from, `recording/trace.py`); and with `--views`, each view as its own clip, frame for
frame (`outside`, `front`, `front_depth`, `hand`, `hand_depth`, `navigation`, `arm_planner`). The floor plan is the scene
release's (`plan/` beside the scene file, [scenes.md](scenes.md#floor-plans)); without one, replay
renders the floor once and keeps it in `~/.cache/phygs/plans/<scene sha256>`.

## Settings a benchmark owns

`configs/environment.yaml` names them in its header; a benchmark passes them as `overrides` and
records them in its own rules file, never a copy of the settings:

| Key | What it decides |
| --- | --- |
| `end_settle_seconds` | How long the robot settles after the stop before the evaluator reads ground truth [s]. |
| `stop_all_timeout_seconds` | How long the stop at the end may take [s]. |
| `contact_record_n` | Arm contacts above this force go into the report [N]. |
| `evaluation_period_seconds` | How often the task samples ground truth while the agent runs (`Task.update`) [s]. |
| `agent_wall_seconds` | The episode ends as `agent_timeout` (the agent's failure) after this much wall time [s]; `null` for no limit. Thinking costs no simulated time on the charged clock, so this bounds a hung agent. |
| `start_check` | The limits of the check before the agent starts: bodies at rest and in place, the robot upright, drawers closed (`drawer_open_max_m: null` lets them start open). |
| `skills.backend` | `collision_avoidance_planner` or `direct`. |
| `planner.backend` | `curobo` or `moveit`. |

## What a benchmark owns, and what it does not

- Owns: the task object and its thresholds, the episode files and their extra blocks, which grants and actions an agent gets (a track), the settings above, how results are aggregated.
- Does not own: the robot, the physics rate, the cameras, the skills, the runtime, the report's keys, how a scene is read. A benchmark that needs one of those changed asks PhyGS for a release.

## Repeats in one World

`World.next_episode(output, seed=...)` gets a World ready for another episode of the same spec in a
fresh folder. The next `run` restores the robot and every scene body to the start (the runtime's
snapshot of the settled scene), clears Nav2's costmaps and goals, settles and runs the start check,
exactly as the first episode did; the arm planner and Nav2 stay up, so a repeat skips Isaac Sim's
start-up (about 40-60 s per episode on 2026-10-29). Each repeat writes its own report, audit, state
log, video and plan requests; the planner's model and log and Nav2's logs stay in the first episode's
folder. A different spec (another scene, robot or object placement) still needs a new World. The robot
test `retrieve_repeat` checks that three repeats in one World start from the same checked state and
succeed as the first does.

What carries over is only what costs no simulated time. The cuRobo planner builds a planner per
gripper angle (about 8 s each); the World warms the measured and open angles before the first
episode, and the arm planner builds any other angle inside one tick before the plan that needs it,
so a repeat is never faster in simulated time for having warm planners. Plans can still differ in
detail between a first episode and a repeat (each planner's random state carries on), as they do
between two processes: compare repeats by their outcomes and times, not plan by plan.

## Versions

A benchmark pins a PhyGS version (`phygs.__version__`, a tag). That version promises the spec schema
(`SCHEMA`) and the report schema (`REPORT_SCHEMA`); every report records both under `schemas`. A
change to either is at least a minor PhyGS release and a benchmark version change. 0.4.0 renamed the
modules a benchmark imports ([layout.md](layout.md#renamed-in-040) has the old and new paths).
