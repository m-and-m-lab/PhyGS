# PhyGS docs

Start here. Each topic has **one** guide. Other files link to that guide; they do not repeat it.

## Start here

| Need | Read |
| --- | --- |
| Install, run, first commands | [README.md](../README.md) |
| How the system fits together: layers, runtime tick, navigation flow (with diagrams) | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| The package layout, the layers and the import rule, what moved in 0.4.0 | [layout.md](layout.md) |
| Driving the framework from a benchmark: spec, task, World, ground truth, report, versions | [world.md](world.md) |
| What is done, what is next, known gaps, decisions | [PLANS.md](../PLANS.md) |
| What changed and when (including every rename of the repository split) | [CHANGELOG.md](../CHANGELOG.md) |
| Exact test and robot-test commands | [tests/README.md](../tests/README.md) · [tests/robot_tests/README.md](../tests/robot_tests/README.md) |
| Rules for coding agents and contributors | [AGENTS.md](../AGENTS.md) |

## Topic guides

| Topic | Guide | Diagram |
| --- | --- | --- |
| Adding a robot: one folder with a `robot.yaml`, plugin points (base controller, actuators), camera roles, the bring-up ladder with `--robot` | [adding_a_robot.md](adding_a_robot.md) | — |
| Observations, frames, timing | [observations.md](observations.md) | — |
| Who sees what: pipelines and permissions | [pipelines.md](pipelines.md) | — |
| Locomotion contract | [locomotion.md](locomotion.md) | Runtime tick diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#runtime-tick-ownership-and-reset) |
| Point navigation (Nav2) | [navigation.md](navigation.md) | Navigation diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#navigation) |
| Nav2 backend, installation (ROS 2 Jazzy, Nav2, explore_lite) and the measured transit envelope | [nav2.md](nav2.md) | Navigation diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#navigation) |
| Mapping (Nav2's) | [mapping.md](mapping.md) | Navigation diagram in [ARCHITECTURE.md](../ARCHITECTURE.md#navigation) |
| Scenes: ids and versions, the scene cache, fetching releases from the dataset | [scenes.md](scenes.md) | — |
| Object search (a robot-test task, not a framework task) | [search.md](search.md) | — |
| Collision-aware manipulation planners (cuRobo, MoveIt): setup, tests, limits | [manipulation_planners.md](manipulation_planners.md) | — |
| Manipulation skills (`stow`, `look_at`, end-effector lines, gripper) and the low-level commands | [manipulation_skills.md](manipulation_skills.md) | — |
| Robot UI (a person as the agent, in a browser, headless simulator) | [ui.md](ui.md) | — |

Scenes are made and released outside PhyGS; PhyGS loads them by id and version ([scenes.md](scenes.md)).

## Keeping the docs tidy

1. **One topic, one guide.** Before you write, find the guide that owns the topic. Link to it; do not copy it.
2. **New guide → add a row here.** A guide that is not listed here is hard to find.
3. **Diagrams of flow or state go in the guide as Mermaid.** GitHub renders them, and they change in the same commit as the text.
4. **Design proposals and their history stay out of this repository.** The docs describe the code as it is; a decision that matters to users goes into the guide that owns the topic and into [CHANGELOG.md](../CHANGELOG.md).
5. **User-visible change → add a line to [CHANGELOG.md](../CHANGELOG.md).**
6. **Superseded content is deleted**, not kept beside the current text. Git keeps the history.
7. **Results go in the guide, not in `output_check/`.** `output_check/` is gitignored; teammates cannot see it.
