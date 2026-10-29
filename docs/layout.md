# Layout: the packages and the import rule

```
phygs/
├── episodes/      The Environment, the session, the spec, the Task seam, scenes by id, recorder, provenance, report   (proposal: world/ + session/)
├── skills/        The public clients: locomotion, navigation, manipulation (api.py each)
├── robots/        Descriptions (robot.yaml), assembly (control.py), one folder per robot
├── execution/     locomotion/ (base controllers), navigation/ (Nav2, footprint, launch), manipulation/ (arm, skills, planners)
├── runtime/       Clock, ownership, expiry, reset, emergency stop
├── simulation/    Isaac Lab adapters (cameras, state, observations) and USD scene reading (ground truth)   (proposal: sim/ + scene/)
├── observations/  Measurement types and localization
├── tools/         Video, panels, timeline, narration, front view, colour detection, collision map   (proposal: recording/ + observations/)
├── scenes/        fetch.py: scene releases from the dataset
├── ui/            The robot UI
├── configs/       environment.yaml
├── agents/        Example agents (proposal: examples/agents)
└── __main__.py    The phygs command
tests/robot_tests/ The robot tests (package phygs_robot_tests): routines, episode plans, components, test tasks, fixture scenes
```

**The rule.** Imports go downward: `ui, examples, tests` → `episodes` → `skills, robots` → `execution` →
`runtime, simulation, observations, tools`. Today five edges go the other way (`execution → robots`,
`execution → skills`, `simulation → execution`, `simulation → robots`, `skills → robots`); the
[layout proposal](design/framework_layout/index.html) cuts them and adds a test that enforces the
rule. Until then, the rule is the intent, not a check.

**Three audiences, three surfaces.** Agents: `phygs.episodes.session` only (tested). Benchmarks:
[world.md](world.md). Framework developers: [adding_a_robot.md](adding_a_robot.md) for a robot;
`execution/*/interfaces.py` or `planners/interface.py` for a backend; [manipulation_skills.md](manipulation_skills.md) for a skill.
