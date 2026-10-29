# The phygs package

The package map, the layers and the import rule are in [docs/layout.md](../docs/layout.md); what each
audience imports:

- **An agent:** `phygs.session` only ([the agent contract](../README.md#-the-agent-contract),
  [examples/agents/template.py](../examples/agents/template.py)).
- **A benchmark:** `phygs.world` and `phygs.scene` ([docs/world.md](../docs/world.md)).
- **A robot:** a folder under `robots/` with a `robot.yaml` ([adding a robot](../docs/adding_a_robot.md)).
- **A backend or a skill:** `execution/*/interfaces.py`, `execution/manipulation/planners/interface.py`,
  [manipulation skills](../docs/manipulation_skills.md), [pipelines](../docs/pipelines.md).

Framework code names no robot's parts (`tests/test_robot_neutral.py`) and imports only downward
(`tests/test_layers.py`). Runnable physical checks are the robot tests,
[tests/robot_tests/](../tests/robot_tests/README.md).
