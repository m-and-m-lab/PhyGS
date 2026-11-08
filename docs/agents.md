# Writing an agent: the session

An agent is a function `run(session)`. The session is the only object it needs ([template](../examples/agents/template.py)):

```python
def run(session):
    print(session.task.goal, session.task.description)      # The public brief; nothing about where the target is.
    packet = session.observe()                              # Cameras (RGB-D, calibrated), joints, IMU, localization.
    goal = session.navigation.navigate_to(x=1.0, y=0.0, heading=0.0)
    session.wait(goal)                                      # Simulated time passes only while the agent waits.
    move = session.manipulation.move_to_pose(pose)          # Planned by cuRobo or MoveIt; feedback has the reason.
    session.wait(move)
    session.manipulation.close_gripper()                    # Holding is measured from contact, not assumed.
    session.finish("done")
```

## Running it

From the Isaac Lab root (`/workspace/isaaclab` in the container), run a world with the template agent
(it looks around and finishes). The grid storage scene is a fixture in the repository, so nothing is
downloaded. The report lands in `--output`:

```bash
PHYGS=scripts/PhyGS
isaaclab -p $PHYGS/examples/run_world.py \
  --spec $PHYGS/tests/robot_tests/configs/episodes/dev_grid_table.yaml \
  --agent template --task_loader phygs_robot_tests.tasks:load_task \
  --output $PHYGS/output_check/$(date +%Y%m%d-%H%M)_grid_table
```

Your own agent is `--agent path/to/agent.py:run`. With the package installed, `phygs <command>` (or
`isaaclab -p -m phygs <command>`) runs the same scripts: `phygs run`, `phygs ui`, `phygs replay`,
`phygs scenes`, `phygs tests`, `phygs version`.

`--task_loader` names the function that turns a spec's task block into a task object (its brief, its
privileged target, its evaluation). PhyGS knows no task; the robot tests ship a retrieve task for
their own use, and a benchmark ships its own. A spec with `task: {goal: none}` needs no loader.

## What the session gives

| Part of the session | What it gives |
| --- | --- |
| `observe()`, `time()`, `wait(command)`, `sleep(seconds)`, `stream(camera)` | Observations and time. The clock is `charged` (time advances only while the agent waits) or `realtime`. |
| `navigation`, `manipulation`, `locomotion` | The clients the spec's `actions` allow (`skills`, `motion` or a list). Each command: an id, `feedback(id)` with status and reason, `cancel(id)`. |
| `grants` and `hints` | Privileged channels the spec grants (`gt_localization`, `known_map`, `object_targets`, `scene_hints`); every use is audited and listed in the report. |
| `finish(reason)` | Ends the episode. The World then cancels everything, keeps the grip, settles and evaluates. |
| `step(name)`, `log(topic, record, images)` | The agent's own account: named steps of its plan on the run's timeline, and records it keeps in the recording (a model's request and reply, with the images it sent). Neither is a call: no time passes and neither is counted. |

The calls of each client: [navigation.md](navigation.md), [locomotion.md](locomotion.md) and
[manipulation_skills.md](manipulation_skills.md). What the cameras give and when:
[observations.md](observations.md).

## Composite skills

The skills that walk and then move the arm are the composite skills (`phygs.skills.composite`, which
imports the session only), given explicit affordances: `skills = CompositeSkills(session)`, then
`skills.open_drawer(part)`, `skills.pick(center, size)`, `skills.place(point)`,
`skills.place_upright(point)`, `skills.close_drawer(part)`. Each walks to where the arm reaches
(stowing first), tries its listed options in order and returns the option that worked and what the
robot measured; the robot tests' plans and ForageBench's reference agents use them. How each works
(stances, options, straight lines): [manipulation_skills.md](manipulation_skills.md#composing-grasps-placements-and-straight-motions).
Where they sit among the public surfaces: [ARCHITECTURE.md](../ARCHITECTURE.md#boundaries).

## The recording and the report

Everything the agent does goes into the episode's recording, `episode.rrd`: a [Rerun](https://rerun.io)
file with the robot, the scene, Nav2, the arm plans and their requests, the agent's calls and the
episode's phases on one timeline (`rerun episode.rrd` opens it; `phygs replay` renders videos from it;
`examples/replay_plans.py` plans the arm plans again). The report (`report.json`) has the status
(`completed`; `setup_error` when the settled world failed the check before the agent started, so the
agent never ran; `error` when the simulator broke), the outcome, the conduct (contacts, drawers left
open), the usage of time and grants, the commands and the provenance. Every file a run leaves, and
replaying it: [world.md](world.md#what-a-run-leaves-and-replaying-it).
