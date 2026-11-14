# The phygs package

The package map, the layers and the import rule are in [docs/layout.md](../docs/layout.md); what each
audience imports:

- **An agent:** `phygs.session` only ([the agent contract](../docs/agents.md),
  [examples/agents/template.py](../examples/agents/template.py)).
- **A benchmark:** `phygs.world` and `phygs.scene` ([docs/world.md](../docs/world.md)).
- **A robot:** a folder under `robots/` with a `robot.yaml` ([adding a robot](../docs/adding_a_robot.md)).
- **A backend or a skill:** `skills/*/execution/interfaces.py`, `skills/manipulation/execution/planners/interface.py`,
  [manipulation skills](../docs/manipulation_skills.md), [pipelines](../docs/pipelines.md).

Framework code names no robot's parts (`tests/test_robot_neutral.py`) and imports only downward
(`tests/test_layers.py`). Runnable physical checks are the robot tests,
[tests/robot_tests/](../tests/robot_tests/README.md).

## The packages

| Package | Holds |
| --- | --- |
| [world/](world/README.md) | `World`, `EpisodeSpec`, the `Task` seam, `Truth`, the report, the robot's assembly (`RobotControl`) |
| [session/](session/README.md) | `AgentSession`, target types, privileged hints |
| [robots/](robots/README.md) | One folder per robot, and the description that loads it |
| [skills/](skills/README.md) | The three clients (locomotion, navigation, manipulation), their execution code, the composite skills |
| [sim/](sim/README.md) | Isaac Lab adapters |
| [recording/](recording/README.md) | The recording, the video and the replay's readers |
| [scene/](scene/README.md) | Scenes by id, the fetcher, the USD reader |
| [runtime/](runtime/README.md) | Clock, ownership, expiry, reset, the observation pipeline |
| [observations/](observations/README.md) | Measurement types, localization, the front view, colour detectors |
| [ui/](ui/README.md) | The robot UI: a person as the agent, in a browser |
| [composite/](composite/README.md) | Deprecated alias of `skills/composite/` |

Up: [repository README](../README.md). `__main__.py` is the `phygs` command; its subcommands are
listed in its docstring ([__main__.py](__main__.py)); the scripts behind `run` and `replay` are in
[examples/README.md](../examples/README.md).
