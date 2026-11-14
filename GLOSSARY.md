# Glossary

The words PhyGS's docs and code use, each with one meaning. Alphabetical within each group. Where a
term has a guide, the link goes there.

## The run

| Term | Meaning |
| --- | --- |
| **Episode** | One run of an agent in a world: from the spec to the report. |
| **Spec** (`EpisodeSpec`) | The episode as one file: scene by id and version, robot and start, objects, an opaque task block, `actions`, `observations`, `grants`, a budget in simulated seconds and a seed. `SCHEMA` numbers its format. [world.md](docs/world.md) |
| **World** (`phygs.world.World`) | Builds an episode from a spec, owns the simulation loop on the main thread, runs the agent, stops, evaluates and writes the report. |
| **Session** (`AgentSession`) | What an agent gets: observations, the skill clients its spec allows, command feedback, time requests, granted hints. The agent's only surface. [agents.md](docs/agents.md) |
| **Agent** | A `run(session)` function (a scripted plan, a planner, a model). It runs in its own thread; its calls are carried out on the main thread. |
| **Runtime** (`EpisodeRuntime`) | The one owner of the clock, every joint, command expiry, reset and the emergency stop. |
| **Tick** | One physics step: 0.004 s (250 Hz). The World steps once per tick; an agent never steps physics. |
| **Clock** | `charged`: simulated time passes only while the agent waits. `realtime`: it always passes. |
| **Budget** | The spec's simulated-time limit; when it runs out the episode ends as `budget`. |
| **Start check** | The check before the agent starts that the settled world is the one the spec describes. If it fails, the episode ends as `setup_error`. |
| **Status** | Whether an episode can be scored: `completed`, `setup_error` (the agent never ran) or `error` (the simulator or the World broke). Only `completed` counts in a rate. |
| **Report** (`report.json`) | What an episode writes: status, how it ended, the task's evaluation, usage, commands, contacts and provenance. Its keys are in `world/report.py`. |
| **Recording** (`episode.rrd`) | A [Rerun](https://rerun.io) file of what flowed in the episode: states, observations, the agent's calls, plans, phases. `phygs replay` renders a run from it. |
| **Provenance** | Hashes of the robot USD, policy files, scene USD and evaluator thresholds, and the code version, recorded in every report. |
| **Run folder** | One run's output folder: `report.json`, `episode.rrd`, logs, and a replay or video when asked. |

## The robot and what it does

| Term | Meaning |
| --- | --- |
| **Robot description** (`RobotDescription`) | Every robot fact the framework reads, from one `robot.yaml` per robot. [adding_a_robot.md](docs/adding_a_robot.md) |
| **Camera role** | A name for a kind of camera in a `robot.yaml` (for example `front`, `planning`), so a spec can run with any robot that has it. |
| **Body frame** | +X forward, +Y left, +Z up, at the robot's root body. The optical frame is +X right, +Y down, +Z forward. The world frame is only exposed through a permitted ground-truth localization. [observations.md](docs/observations.md#coordinates-and-information-sources) |
| **Front view** | The robot's two front cameras joined into one forward-looking image (`observations/front_view.py`). |
| **Client** | One of the three public skill interfaces on a robot: `locomotion`, `navigation`, `manipulation`. Each is a thin `api.py` over an `execution/` backend. |
| **Skill** | A client method that does a whole step, such as `navigate_to`, `explore`, `stow`, `unstow` and `look_at`. [manipulation_skills.md](docs/manipulation_skills.md) |
| **Composite skill** | Agent-side code over the session that walks, then moves the arm: `pick`, `place`, `place_upright`, `open_drawer`, `close_drawer` (`phygs.skills.composite`). |
| **Backend** | The machinery behind a client: the base controllers, Nav2, the arm controller, the planners (cuRobo, MoveIt). |
| **Command** | A request to a client. It returns an id; `feedback(id)` reports `active`, `stopping`, `succeeded`, `failed`, `cancelled` or `overridden` (replaced by a later command); `cancel(id)` stops it. |
| **Actions** | The spec's list of clients an agent may use: the sets `skills` (navigation, manipulation) and `motion` (those plus locomotion). |
| **Transit footprint** | The measured outline and sweep of the robot walking with the arm stowed, which navigation plans with. [nav2.md](docs/nav2.md) |

## Information and measurement

| Term | Meaning |
| --- | --- |
| **Grant** | A privileged channel a spec gives an agent: `gt_localization`, `known_map`, `object_targets`, `scene_hints`. Every use is audited and listed in the report. |
| **Hint** | A privileged fact an agent reads through `session.hints`; refused without its grant. |
| **Target** | The object the privileged channels (hints, known map, arm planner) treat as the one asked for; the caller passes it to the World, never the framework. |
| **Observation pipeline** | The capture, permissions, views and processors between the sensors and an agent. [pipelines.md](docs/pipelines.md) |
| **Known map** | The scene's boxes given to navigation and the arm planner, as an alternative to mapping from depth (grant `known_map`). |
| **Truth** (`Truth`) | Ground-truth measurements of a running world: poses, openings, contacts, how far an object was lifted. Read only by the task evaluator and the robot tests, never by an agent. |
| **Task** | The object a benchmark passes to the World: a public goal and description, a brief, and the evaluator's `reset`, `update` and `result`. PhyGS never imports one. |
| **Task evaluator** | The part of a task that decides the result from `Truth`, after the robot has stopped. Skill feedback is never the result. |
| **Conduct** | What the evaluator records beside the outcome: unintended contacts, drawers left open. Which ones a benchmark penalizes is its rule. |
| **Track** | A benchmark's choice of which grants and actions an agent gets. |
| **Scene** | A USD house or room, named by id and version. Scenes are made and released outside PhyGS. [scenes.md](docs/scenes.md) |
| **Built-in scene** | A small scene that ships with PhyGS (`grid_storage`, `nav_course`, `dynamic_course`, in `phygs/scene/builtin/`), used by the robot tests. |

## Tests

| Term | Meaning |
| --- | --- |
| **Robot test** | A test that spawns a scene and has the robot do one thing, then checks named values against thresholds (`check.json`). |
| **Rung** | One robot test's place on the ladder, in the order the framework is built. [tests/robot_tests/README.md](tests/robot_tests/README.md) |
| **Ladder** (`tests/robot_tests/ladder.json`) | All the rungs in build order, with their scripts, requirements, where their thresholds are and last evidence. The first failing rung is the lowest broken module. |
| **Agent test** | A rung written the way an agent would use the framework: an episode spec, a plan through the session, checks from ground truth. (`tests/robot_tests/episodes/`) |
| **Component test** | A rung for machinery under the session (stage and reset, cameras, gait, footprint), driving the modules directly. (`tests/robot_tests/components/`) |
| **Suite** | The list `run_suite.py` runs: the whole ladder or a group of it. |
