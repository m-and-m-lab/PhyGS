# PhyGS docs

Start here. Each topic has **one** guide. Other files link to that guide; they do not repeat it.
Design pages under `design/` explain the parts that are hard to follow in text and record the
decisions behind them.

## Start here

| Need | Read |
| --- | --- |
| Install, run, first commands | [README.md](../README.md) |
| How the system fits together: layers, runtime tick, navigation flow (with diagrams) | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| The package layout, module by module, and the import rule | [layout.md](layout.md) · [phygs/README.md](../phygs/README.md) |
| Driving the framework from a benchmark: spec, task, Environment, ground truth, report, versions | [world.md](world.md) |
| What is done, what is next, known gaps, decisions | [PLANS.md](../PLANS.md) · [status board](design/visual/status.html) |
| What changed and when (including every rename of the repository split) | [CHANGELOG.md](../CHANGELOG.md) |
| Exact test and robot-test commands | [tests/README.md](../tests/README.md) · [tests/robot_tests/README.md](../tests/robot_tests/README.md) |
| Rules for coding agents and contributors | [AGENTS.md](../AGENTS.md) |
| Syncing changes from the lab's development tree (interactive-search) | `tools/sync_from_interactive_search.py --help` |

## Topic guides

| Topic | Guide | Visual |
| --- | --- | --- |
| Adding a robot: one folder with a `robot.yaml`, plugin points (base controller, actuators), camera roles, the bring-up ladder with `--robot` | [adding_a_robot.md](adding_a_robot.md) | [Design and status](design/robot_portability/index.html) |
| Observations, frames, timing | [observations.md](observations.md) | [Frames and conventions](design/visual/frames.html) |
| Who sees what: pipelines and permissions | [pipelines.md](pipelines.md) | [Who sees what](design/visual/pipelines.html) |
| Locomotion contract | [locomotion.md](locomotion.md) | Runtime tick diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#runtime-tick-ownership-and-reset) |
| Point navigation (Nav2) | [navigation.md](navigation.md) | Navigation diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#navigation) |
| Nav2 backend, installation (ROS 2 Jazzy, Nav2, explore_lite) and the measured transit envelope | [nav2.md](nav2.md) | Navigation diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#navigation) |
| Mapping (Nav2's) | [mapping.md](mapping.md) | Navigation diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#navigation) |
| Scenes: ids and versions, the scene cache, fetching releases from the dataset, publishing a new version | [scenes.md](scenes.md) | — |
| Object search (a robot-test task, not a framework task) | [search.md](search.md) | — |
| Collision-aware manipulation planners (cuRobo, MoveIt): setup, tests, limits | [manipulation_planners.md](manipulation_planners.md) | [status board](design/visual/status.html) |
| Manipulation skills (`stow`, `look_at`, end-effector lines, gripper) | [manipulation_skills.md](manipulation_skills.md) | [Manipulation contract](design/manipulation_contract/index.html) |
| Robot UI (a person as the agent, in a browser, headless simulator) | [design/ui/design.html](design/ui/design.html) | Wireframe and Front-view evidence in the same page |

## Design pages

Plain HTML in git, no build step. GitHub shows HTML as source, so open them locally:

```bash
# From the PhyGS checkout. Then open http://localhost:8000/index.html (VS Code forwards the port).
python3 -m http.server 8000 --directory docs/design/episode_harness
```

| Page | What it records |
| --- | --- |
| [episode_harness/](design/episode_harness/index.html) | The entry point, the session an agent gets, actions and grants, the spec, the evaluator slot, the robot-UI session. Implemented. |
| [manipulation_contract/](design/manipulation_contract/index.html) | Joint and end-effector trajectories, guards, what agents compose. Implemented 2026-10-18. |
| [robot_portability/](design/robot_portability/index.html) | One folder per robot, roles instead of names, the bring-up ladder per robot; Ridgeback-Franka as the second robot. Implemented 2026-10-28. |
| [evaluator_and_robot_tests/](design/evaluator_and_robot_tests/index.html) | How a task evaluator reads ground truth and how the robot tests form a numbered ladder. |
| [framework_benchmark_split/](design/framework_benchmark_split/index.html) | What belongs to the framework and what to a benchmark; the one `Task` seam between them. The seam is implemented (`phygs/episodes/task.py`). |
| [ui/](design/ui/design.html) | The robot UI. |
| [visual/](design/visual/status.html) | Status board, pipelines, frames. |
| [framework_layout/](design/framework_layout/index.html) | Proposal (2026-10-29, not implemented): the layout PhyGS should have as a framework (session/world/sim/scene/recording layers, imports downward only, one `phygs` command, agents as examples), the module mapping, and five steps with deprecations. |
| [archive/](design/archive/robot_tests_housekeeping/index.html) | Superseded designs kept for the record. |

## Scene generation

The Infinigen fork and its notes are under [scene_generation/](../scene_generation/): the generator
(`infinigen/`, its own Python environment), the USD builders, packager and publisher (`usd_tools/`)
and the asset catalogs and legacy notes (`docs/`). Fetching and publishing released scenes: [scenes.md](scenes.md).

## Keeping the docs tidy

1. **One topic, one guide.** Before you write, find the guide that owns the topic. Link to it; do not copy it.
2. **New guide → add a row here.** A guide that is not listed here is hard to find.
3. **Diagrams of flow or state go in the guide as Mermaid.** GitHub renders them, and they change in the same commit as the text.
4. **Design pages go in `docs/design/`.** Each page names the date it was checked. When the code changes, update the page or mark it stale.
5. **User-visible change → add a line to [CHANGELOG.md](../CHANGELOG.md).**
6. **Superseded content moves to a clearly marked section** or to `design/archive/`. Do not leave two current-sounding versions.
7. **Results go in the guide, not in `output_check/`.** `output_check/` is gitignored; teammates cannot see it.
