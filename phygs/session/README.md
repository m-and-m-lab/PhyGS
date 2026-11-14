# session

What an agent gets. Agents import from `phygs.session` only.

| File | Holds |
| --- | --- |
| `session.py` | `AgentSession`: the action-filtered skill clients, `observe()` and `stream()`, requests for simulated time, the audit of every call; the target types (`ObjectTarget`, `PixelTarget`, `PointTarget`), trajectories, `Command`, and the errors an agent catches (`EpisodeEnded`, `SkillFailure`). |
| `hints.py` | Privileged hints: ground truth from the simulation, each call refused without its grant. |

The agent runs `run(session)` in its own thread; every call is carried out on the main thread by
the World between ticks. No Isaac Sim imports. The agent contract:
[docs/agents.md](../../docs/agents.md); a starting agent:
[examples/agents/template.py](../../examples/agents/template.py).

Up: [phygs/README.md](../README.md). Tests: `tests/test_agent_session.py`,
`tests/test_agent_imports.py`, `tests/test_public_surface.py`.
