# ui

The robot UI: a person runs the robot from a browser while the simulation runs headless. The
person is the agent, so every action is an audited session call within the spec's actions and grants.

| Path | Holds |
| --- | --- |
| `cli.py` | The entry point (`phygs ui`): builds a `World` with the UI as agent and viewer. |
| `session.py` | `UiSession`: the page's requests become session calls; the episode's read-only view becomes frames, map, state and step lines. |
| `server.py`, `bridge.py` | The page's HTTP and WebSocket server (its own thread) and the thread-safe exchange of plain data with the runner thread. |
| `translate.py`, `targets.py`, `schema.py`, `codecs.py` | What the person chose on the page as the arguments a skill already takes; what a pointed pixel is; the skills the page can run; image encoding. |
| `web/` | The page: `index.html`, `app.js`, `style.css`. |
| `configs/` | The UI's episode specs: `episode.yaml`, `grid_drawer.yaml`, `house_drawer.yaml`. |

What the page shows, and what it never shows: [docs/ui.md](../../docs/ui.md).

Up: [phygs/README.md](../README.md). Tests: `tests/test_ui_*.py` (with `tests/ui_fakes.py`).
