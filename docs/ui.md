# The robot UI

A person runs the robot from a browser while the simulation runs headless. The person is the agent:
every action is an audited session call within the spec's actions and grants, like any agent's.

```bash
isaaclab -p scripts/PhyGS/phygs/ui/cli.py --task_loader phygs_robot_tests.tasks:load_task \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_ui_grid
```

Open the printed link (it carries a token). From another computer, forward the port first:
`ssh -L 8765:127.0.0.1:8765 <host>`. One browser at a time; a second connection is refused.
`phygs/ui/configs/house_drawer.yaml` runs the Bear house; `--robot` overrides the spec's robot;
`--demo tour` drives a short route by itself so there is something to watch.

## How it works

`phygs/ui/cli.py` builds a `World(spec, clock="realtime", viewer=ui, hold_clock=True)` and runs
`UiSession.agent` as its agent (`phygs/ui/session.py`). The page's requests become session calls on the
agent thread. `UiSession.update` is the World's viewer: once per camera period it publishes frames, the
map, the robot state and the skill step lines from the read-only `EpisodeView`. The World builds the
robot, the cameras, Nav2, the arm planner and the skills, evaluates the end state and writes the
report; the UI builds none of them. With `hold_clock`, simulated time starts when the person presses
Start.

## What the page shows, and what it never shows

| Shown | Never shown |
| --- | --- |
| The robot's front cameras (joined into one Front view when there are two) and its look (hand) camera, RGB and depth | The third-person camera and the viewport (recorded for reviewers only) |
| The robot's pose, joints, gripper and whether it holds something | Object poses and names, drawer parts, the known-box map |
| Navigation's own map: Nav2's global costmap | The planned navigation path (the agent API does not return it) |
| Skill feedback, step lines, the task text | |

A unit test serializes every message type and fails if a privileged field appears.

**Front view.** One virtual level camera at the midpoint of the two front cameras, looking along the
body's +x, 110° wide. Each pixel is looked up in the camera that sees it nearer its image centre, at
the depth that camera measured, so furniture lines up across the seam and every pixel keeps its 3D
point. Pixels neither camera saw are black.

## Running a skill

Left: skill tiles. Centre: the view (tabs, with the map in a corner) and the command bar of the
selected skill. Right: one card per channel (base, arm), the robot state and the log. Bottom: a
timeline of every command.

1. Click a tile (or press its number): it is armed, and the view switches to the picker it needs
   (Navigate to opens the map; Look at and Move to pose open the last camera view).
2. Click the target and set the other inputs. A click on an image is turned into a world point with
   the frame the person saw (the nearest valid depth within 5 px); the bar shows the point, its
   distance and the surface normal.
3. Press Run or Enter. The tile, its card and a timeline block show it running, then ✓ or ✗ with the
   raw failure code and reason.

Keys: 1–9 select a tile, Enter runs, Esc clears, W A S D Q E drive while Drive is armed, T opens the
telemetry drawer. A running command is cancelled from its card. E-stop stops the base, cancels the
arm's commands and finishes the episode. End asks for a second click within 3 s.

The tiles cover every public call: Navigate to, Drive, Stop, Look at, Arm joints, Move to pose, Move
along (a wrist line), Joint path, Stow, Unstow, Gripper open and close, and the cancels. The server
sends the tiles and their inputs as a schema when the page connects (`phygs/ui/schema.py`), so a new
API call needs one schema entry, not new page code.

**Map.** Free (white), unknown (grey, Nav2's cost 255), occupied (dark), too close for the robot's
centre (pink, Nav2's inscribed and lethal cells), the robot, its trail and the goal. Click sets the goal,
drag sets the heading.

## What a run leaves

In the output folder: the World's `report.json` and `episode.rrd`, as for every agent;
`ui_summary.json` (commands and failures per skill, Drive time, the evaluation); `url.txt` (the
page's link) and `timing.json` (times, map updates, how it ended). The person's episode
is recorded like any agent's: `rerun episode.rrd` shows it, and `phygs replay` renders its videos.
The recording also keeps the page's side, as the UI's own records (`session.log`, from the agent's
first pass until the recording closes): each request the UI received (`/agent/log/ui/request`: the
page's, and the launcher's own, autostart, Ctrl+C or SIGTERM and the time limit, which carry
`source`) and each event the page was told, with its number (`/agent/log/ui/event`: command changes with their raw
feedback, previews, step lines, errors, the episode's start and end); `Recording.agent_log("ui/event")`
reads them. The result card shown after the episode is in `ui_summary.json`. The telemetry drawer's
data is recorded where it comes from: the robot's pose and its base link's measured velocity at every
state sample (each camera period), the joints' positions, velocities and efforts every physics tick, the commands'
feedback among the agent calls, each arm plan's request and result under `/planner`, and the
real-time factor and tick time under `/perf`. The drawer's hand position follows from the base
pose, the joints and the robot's hand tool, and its camera rates from the capture settings (the recording's `observation` property); neither is
recorded.

## Protocol

A WebSocket between the page and `phygs/ui/server.py`. The server sends `hello` (the task, the skill
schema, the views, the map, the limits), `state` (about 10 Hz wall: times, robot state, active
commands), binary `frame` messages and events; the page sends skill requests, clicks
(`{view, stamp, x, y}`) and cancels. Old frames are dropped, so a slow browser never slows the
simulation. The message types and their fields are in `phygs/ui/server.py` and `phygs/ui/schema.py`.
