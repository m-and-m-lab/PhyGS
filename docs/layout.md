# Layout: the packages and the import rule

```
phygs/
├── ui/            The robot UI: a person as the agent (server, bridge, page session, cli.py)
├── world/         What a benchmark drives: World (build, run, stop, evaluate, report), EpisodeSpec, the Task
│                  seam, Truth, the report schema, recorder, provenance, the Nav2 assembly, the viewer protocol
├── session/       What an agent gets: AgentSession, its target types and errors, privileged hints
├── robots/        One folder per robot (robot.yaml), the description that loads it, its scene, its assembly (control.py)
├── skills/        The public clients: locomotion, navigation, manipulation (api.py each)
├── execution/     locomotion/ (base controllers), navigation/ (Nav2, footprint, known map, launch),
│                  manipulation/ (arm, skills, planners)
├── sim/           Isaac Lab adapters: cameras, robot state and observations, base and arm readers, footprint, Fabric
├── recording/     The video, its panels, the command timeline, the skills' narration
├── scene/         Scenes by id (registry), the fetcher, the USD reader (ground truth), drawers, storage units, plain types
├── runtime/       Clock, ownership, command expiry, reset, emergency stop
├── observations/  Measurement types, localization, the front view, the reference colour detectors
├── configs/       environment.yaml (every setting, the benchmark-owned ones named), navigation/, ui/, robots/, scenes/
└── __main__.py    The phygs command
examples/          run_world.py (phygs run) and agents/ (template, depth_example)
tests/             Unit tests; robot_tests/ is the robot-test ladder (package phygs_robot_tests)
```

## The rule: imports go downward

The packages are layers, listed top to bottom; a module imports its own package and any package
in a lower layer, never one beside or above it:

| Layer | Packages | Imports |
| --- | --- | --- |
| 1 | `ui` | everything below |
| 2 | `world` | the session, robots, skills, execution, and the bottom layers |
| 3 | `session` | robots (`RobotInfo`), execution (the target types it re-exports) |
| 4 | `robots` | skills (the clients it assembles), execution, sim, recording, runtime, observations |
| 5 | `skills` | execution, runtime |
| 6 | `execution` | recording (the planner's panel), scene (the known map, the part type), runtime, observations |
| 7 | `sim`, `recording` | scene (scene bodies), runtime, observations |
| 8 | `scene`, `runtime` | observations (runtime only) |
| 9 | `observations` | nothing in PhyGS |

`tests/test_layers.py` enforces it: every package has a layer, no module imports beside or above
its own. Imports under `if TYPE_CHECKING:` are
annotations, not dependencies, and are not checked: a lower layer is sometimes handed an upper
layer's object and names its type (a base controller is built from the robot's description; a
skill backend drives the client it serves; navigation leases the locomotion client). Three such
names remain: `RobotDescription`, `ManipulationClient` and `LocomotionClient`.

Why this order: the robot's assembly (`robots/control.py`) and the world's (`world/`) are the two
places that know everything, so they sit on top of what they assemble; everything under them can be
read, tested and replaced without them. A new package gets a row here and in `LAYERS`.

## Three audiences, three surfaces

- **Agent authors:** `phygs.session` only (`tests/test_agent_imports.py`), with the standard
  library, NumPy and PyYAML. [The agent contract](../README.md#-the-agent-contract);
  `examples/agents/template.py`.
- **Benchmark authors:** `phygs.world` and `phygs.scene` (scenes by id). [world.md](world.md).
- **Framework developers:** a robot is a folder ([adding_a_robot.md](adding_a_robot.md)); a
  backend implements `execution/*/interfaces.py` or `execution/manipulation/planners/interface.py`;
  a skill is a client method with a backend behind it ([manipulation_skills.md](manipulation_skills.md)).

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
| `phygs.simulation.maps` | `phygs.execution.navigation.known_map` |
| `phygs.simulation.navigation` | `phygs.world.navigation` |
| `phygs.tools.{front_view, color_detection}` | `phygs.observations.{front_view, color_detection}` |
| `phygs.tools.{video, panels, timeline, narration}` | `phygs.recording.{...}` (same names) |
| `phygs.tools.collision_map` | `phygs.execution.manipulation.collision_map` |
| `phygs.agents.{template, depth_example}` | `examples/agents/{template, depth_example}.py` (`--agent template` unchanged) |

`ArticulatedPart` is defined in `phygs.scene.types` now; the skills' types and the session re-export
it, so `from phygs.session import ArticulatedPart` works.
