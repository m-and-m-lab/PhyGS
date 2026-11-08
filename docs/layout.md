# Layout: the packages and the import rule

```
phygs/
├── ui/            The robot UI: a person as the agent (server, bridge, page session, cli.py; configs/: its specs)
├── world/         What a benchmark drives: World (build, run, stop, evaluate, report), EpisodeSpec, the Task
│                  seam, Truth, the report schema, recorder, provenance, the Nav2 assembly, the viewer protocol,
│                  the replay; the robot's assembly (robot.py: RobotControl); environment.yaml (every setting,
│                  the benchmark-owned ones named)
├── session/       What an agent gets: AgentSession, its target types and errors, privileged hints
├── robots/        One folder per robot (robot.yaml) and the description that loads it
├── skills/        Every skill. One folder per client, each its api.py and its execution/:
│                  locomotion/ (base controllers), navigation/ (Nav2, footprint, known map, launch/ with
│                  Nav2's parameters), manipulation/ (the arm controller, the stow and look_at skills, planners);
│                  and composite/: the skills that walk, then move the arm (pick, place, place_upright, open and
│                  close drawers, the look_at sweep), agent-side code over the session
├── composite/     Deprecated alias of skills/composite/ (moved 2026-11-05)
├── sim/           Isaac Lab adapters: cameras, robot state and observations, base and arm readers, footprint,
│                  Fabric, the robot's Isaac Lab scene (robot_scene.py)
├── recording/     The episode's recording and its reader, the video, its panels, the command timeline,
│                  the skills' narration, the trace and the replay's overview strip
├── scene/         Scenes by id (registry), the fetcher, the USD reader (ground truth), drawers, storage units, floor
│                  plans, plain types, the built-in scenes (builtin/)
├── runtime/       Clock, ownership, command expiry, reset, emergency stop
├── observations/  Measurement types, localization, the front view, the reference colour detectors
└── __main__.py    The phygs command
examples/          run_world.py (phygs run), replay_run.py (phygs replay) and agents/ (template, depth_example)
tests/             Unit tests; robot_tests/ is the robot-test ladder (package phygs_robot_tests)
```

## The rule: imports go downward

The packages are layers, listed top to bottom; a module imports its own package and any package
in a lower layer, never one beside or above it:

| Layer | Packages | Imports |
| --- | --- | --- |
| 1 | `ui` | everything below |
| 2 | `world`, `skills/composite` (`composite`: its deprecated alias) | `world`: the session, robots, skills (the clients it assembles), and the bottom layers; `skills/composite`: the session only (agent-side code, a layer of its own inside `skills/`) |
| 3 | `session` | robots (`RobotInfo`), skills (the target types it re-exports) |
| 4 | `robots` | skills (the planning model of a robot) |
| 5 | `skills` | sim (the render GPU), recording (the planner's panel), scene (the known map, the part type), runtime, observations |
| 6 | `sim`, `recording` | scene (scene bodies), runtime, observations |
| 7 | `scene`, `runtime` | observations (runtime only) |
| 8 | `observations` | nothing in PhyGS |

`tests/test_layers.py` enforces it: every package has a layer, no module imports beside or above
its own. Imports under `if TYPE_CHECKING:` are
annotations, not dependencies, and are not checked: a lower layer is sometimes handed an upper
layer's object and names its type (a base controller is built from the robot's description; a
skill backend drives the client it serves; navigation leases the locomotion client). Three such
names remain: `RobotDescription`, `ManipulationClient` and `LocomotionClient`.

Why this order: the World (`world/`, with the robot's assembly, `world/robot.py`) is the place that
knows everything, so it sits on top of what it assembles; everything under it can be read, tested
and replaced without it. A folder sits inside the one package that uses it (a skill's execution,
a config beside its reader); what several packages use stays at the top. A new package gets a row
here and in `LAYERS`.

## Three audiences, three surfaces

- **Agent authors:** `phygs.session` only (`tests/test_agent_imports.py`), with the standard
  library, NumPy and PyYAML, and optionally the composite skills over it (`phygs.skills.composite`, which
  imports the session only). [The agent contract](../README.md#-the-agent-contract);
  `examples/agents/template.py`.
- **Benchmark authors:** `phygs.world` and `phygs.scene` (scenes by id). [world.md](world.md).
- **Framework developers:** a robot is a folder ([adding_a_robot.md](adding_a_robot.md)); a
  backend implements `skills/*/execution/interfaces.py` or `skills/manipulation/execution/planners/interface.py`;
  a skill is a client method with a backend behind it ([manipulation_skills.md](manipulation_skills.md)).

## Moved in 0.10.0

Each skill's machinery moved under its client, each config beside its one reader, and the robot
assembly out of `robots/`. Code that imports an old path fails with `ModuleNotFoundError` and needs
the new one (ForageBench imports none of them):

| Old | New |
| --- | --- |
| `phygs.execution.{locomotion, navigation, manipulation}` | `phygs.skills.{locomotion, navigation, manipulation}.execution` |
| `phygs.execution.manipulation.skills.{backends, runner, types}` | `phygs.skills.manipulation.execution.{backends, runner, types}` |
| `phygs.robots.control` (`RobotControl`) | `phygs.world.robot` |
| `phygs.robots.scene` | `phygs.sim.robot_scene` |
| `phygs/configs/environment.yaml` | `phygs/world/environment.yaml` |
| `phygs/configs/navigation/{nav2.yaml, spot_nav2.rviz}` | `phygs/skills/navigation/execution/launch/` |
| `phygs/configs/ui/*.yaml` | `phygs/ui/configs/` |
| `phygs.robots.posture`, `ArmClient`, `session.arm`, `actions: joints` | removed: `manipulation.move_joints` moves arm joints |

## Renamed in 0.4.0

The old paths were removed in 0.4.0 itself, without a deprecation period: code that imports one
fails with `ModuleNotFoundError` and needs the new path below (the [CHANGELOG](../CHANGELOG.md) has
the same table).

| Old | New |
| --- | --- |
| `phygs.episodes.session`, `.hints` | `phygs.session.session`, `.hints` (agents: `from phygs.session import ...`) |
| `phygs.episodes.environment` (`Environment`) | `phygs.world.world` (`World`; `from phygs.world import World`) |
| `phygs.episodes.{spec, task, viewer, recorder, provenance, report}` | `phygs.world.{spec, task, viewer, recorder, provenance, report}` |
| `phygs.episodes.scenes`, `phygs.scenes.fetch` | `phygs.scene.registry`, `phygs.scene.fetch` |
| `phygs.simulation.{cameras, fabric, observations, base, arm, footprint}` | `phygs.sim.{...}` (same names) |
| `phygs.simulation.world` | `phygs.sim.state` |
| `phygs.simulation.{scene, parts, scene_assets, storage_units}` | `phygs.scene.{reader, parts, assets, storage_units}` |
| `phygs.simulation.maps` | `phygs.skills.navigation.execution.known_map` |
| `phygs.simulation.navigation` | `phygs.world.navigation` |
| `phygs.tools.{front_view, color_detection}` | `phygs.observations.{front_view, color_detection}` |
| `phygs.tools.{video, panels, timeline, narration}` | `phygs.recording.{...}` (same names) |
| `phygs.tools.collision_map` | `phygs.skills.manipulation.execution.collision_map` |
| `phygs.agents.{template, depth_example}` | `examples/agents/{template, depth_example}.py` (`--agent template` unchanged) |

`ArticulatedPart` is defined in `phygs.scene.types` now; the skills' types and the session re-export
it, so `from phygs.session import ArticulatedPart` works.
