# The world: what a benchmark drives

A benchmark says what is asked and how it is scored; PhyGS runs the robot in the world and measures
what happened. This page is the benchmark author's: build a world from a spec, pass a task, run an
agent, read the report. Everything here is `phygs.world` (and `phygs.scene` for scenes by id).

## One run, in code

```python
from phygs.world import EpisodeSpec, World, load_agent   # World after the Isaac Sim app has started

spec = EpisodeSpec.load("episode.yaml")                  # validated before Isaac Sim starts; scene by id
task = my_benchmark.load_task(spec.task)                 # brief, reset/update/result, provenance; keeps its answer
name, agent = load_agent("examples/agents/template.py:run")  # any run(session) function
with World(spec, output_folder, task=task, target="box", clock="charged",  # target: the privileged channels' object
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
| Spec | `world/spec.py` (`EpisodeSpec`, `SCHEMA = 3`, `load_agent`) | Scene by id and version, robot and start, objects (`{name: {position, frame?}}`), drawers open at the start (`parts: {name: {opening: m}}`; what lies inside rides along), an opaque `task` block, actions, observations, grants, budget, seed. A benchmark adds its own blocks beside these and strips them before `from_dict`. |
| Task | `world/task.py` (`Task` protocol, `NoTask`, `load_task(block, loader)`) | `goal`, `description` (public), `brief()`, `reset(truth)` (the settled world, before the agent starts), `update(truth)` (every `evaluation_period_seconds` while the agent runs), `result(truth, end)` (once, after the stop and settle; `end` is an `EpisodeEnd`: how it ended, the simulated time, the arm contacts labelled unintended and the grasp's squeeze, for the benchmark's own conduct rules), `provenance()`. The framework never imports a task; it takes the object, and never reads the task's answer: the caller passes `World(target=...)`, the object the hints, the known map and the arm planner treat as the target; the grants decide what the agent learns. |
| World | `world/world.py` (`World`, `run_episode`) | Builds the scene, robot, runtime, backends; checks the settled world before the agent starts (`world/start.py`); runs the agent in a thread; audits; stops; evaluates; writes the report. `overrides` pins settings over `world/environment.yaml` (`world/build.py`, `SETTINGS`). |
| Ground truth | `World.truth()` → `Truth` (`world/truth.py`) | Object poses and boxes, `held`, `pinched` (every jaw pressing on a body), arm contacts, the drawer a body started in, how far bodies moved from where they were placed, drawer openings, base pose, arm joints, stow error. Only the task's evaluator and robot tests read it. |
| Report | `world/report.py` (`REPORT_SCHEMA`, `KEYS`, `compose`), `world/recorder.py`, `world/provenance.py` | `report.json` keys with their meaning; the video; hashes of every asset and threshold and the code version. `status` says whether the episode can be scored: `completed`, `setup_error` (the start check failed, the agent never ran) or `error` (the simulator or the World broke). Only `completed` episodes count in a rate. |
| Scenes | `scene/registry.py` (`scene_file(id, version)`), `scene/fetch.py` (`phygs scenes`) | Roots: `$PHYGS_SCENES`, `~/.cache/phygs/scenes`, the fixtures. [scenes.md](scenes.md). |
| Session | `phygs.session` (`AgentSession`, target types) | What the agent gets; a benchmark's agents import only this. |

## What a run leaves, and replaying it

A run's output folder holds:

| File | What it is |
| --- | --- |
| `report.json` | The report (`world/report.py`). Written at the episode's start with `status: running`, the run's id, the spec, the provenance and where the recording is, and replaced by the full report at the end, so a run that dies still says what ran. |
| `episode.rrd` | The episode's recording (`recording/log.py`), a [Rerun](https://rerun.io) file named by the report's `run_id`, of what flowed in the episode. Every physics tick: the joint positions, the joint targets sent and the applied efforts. Every camera period: the robot's base pose (its URDF's collision geometry once), whether the gripper holds something and the base's commanded velocity; every scene body's pose when it moved more than 1 mm or 0.25° (its collider boxes once). Every contact read: each arm contact's force (0 when it ends). Each global costmap Nav2 publishes and each route change; each arm plan's request as the planner worker got it, with its depth images, and its outcome with the planned trajectory (`examples/replay_plans.py` plans them again offline), and its display; each observation the agent received, with each camera image once, lossless; every agent call (time, call, arguments, result or error) and the agent's named steps (`with session.step("open the drawer"):` logs `session.step` lines where the step begins and ends, with the error that left it; steps nest and are not counted in the report's `usage`); each command span when it ends; the episode's phases (settle, start check, agent, stop, end settle, evaluation) and anomalies; and the real-time factor, tick time and machine load once per second. Its properties: the spec, the provenance, the run's settings and the cameras' capture configuration. What is true lives under `/truth`, apart from what the agent could know. `recording/reader.py` (`Recording`) reads it back. `recording.level` in `world/environment.yaml` (or `run_world.py --recording`): `full`, `all_frames` (also every camera capture as the cameras gave it: about 5 GB per simulated minute with 3 cameras) or `report_only`. Logging never fails or changes the episode: its first error closes the recording, and the report's `recording` entry says `failed` with the error. The file is made safe on disk at every phase; a run that dies still reads back to the last whole message. `rerun episode.rrd` opens it in the episode's layout (`recording/blueprint.py`); `recording.live` in `world/environment.yaml` also streams it to a viewer you started: `rerun` with no arguments listens on port 9876, so `live: rerun+http://127.0.0.1:9876/proxy`, through `ssh -R 9876:localhost:9876` when the run is on another machine. |
| `third_person.mp4` | With `--record` only: the video, rendered while the run goes (it slows the run), in the same look as `phygs replay` (`recording/video.py`): a camera following the robot, the panels on its right, the overview strip under both with the timeline so far. |
| `nav2/`, `planner/` | The Nav2 servers' and the arm planner worker's logs. |

`phygs replay <run>` (`examples/replay_run.py`) renders the run afterwards from `report.json` and `episode.rrd`, with the run's settings: it builds
the scene and robot again without control, Nav2 or the planner, poses them sample by sample without
stepping physics, renders the robot's cameras and makes their images with the run's own camera code
and capture configuration (a recording at level `all_frames` holds every capture: replay shows those
instead), and writes into `<run>/replay`: `replay.mp4`, laid out like the `--record` video (a
camera following the robot, with the recording's panels on its right: front and hand cameras, their
depth, navigation with Nav2's costmap and route, and the arm planner), with the overview strip under it
(clock, what the gripper holds, the arm's contact force at that time, the agent's step, the floor plan with the path walked and
Nav2's route, and the timeline of steps, commands, grips and bumps); `trace.json` (the timeline and path
the strip is drawn from, `recording/trace.py`); and with `--views`, each view as its own clip, frame for
frame (`outside`, `front`, `front_depth`, `hand`, `hand_depth`, `navigation`, `arm_planner`). The floor plan is the scene
release's (`plan/` beside the scene file, [scenes.md](scenes.md#floor-plans)); without one, replay
renders the floor once and keeps it in `~/.cache/phygs/plans/<scene sha256>`.

## Settings a benchmark owns

`world/environment.yaml` names them in its header; a benchmark passes them as `overrides` and
records them in its own rules file, never a copy of the settings:

| Key | What it decides |
| --- | --- |
| `end_settle_seconds` | How long the robot settles after the stop before the evaluator reads ground truth [s]. |
| `stop_all_timeout_seconds` | How long the stop at the end may take [s]. |
| `contact_record_n` | Arm contacts above this force go into the report [N]. |
| `evaluation_period_seconds` | How often the task samples ground truth while the agent runs (`Task.update`) [s]. |
| `agent_wall_seconds` | The episode ends as `agent_timeout` (the agent's failure) after this much wall time [s]; `null` for no limit. Thinking costs no simulated time on the charged clock, so this bounds a hung agent. |
| `start_check` | The limits of the check before the agent starts: bodies at rest and in place, the robot upright, each drawer within `drawer_open_max_m` of the spec's opening (closed unless `parts` opens it; `null` leaves drawers unchecked). |
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
folder. A different spec (another scene, robot, object placement or drawer opening) still needs a new World. The robot
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
