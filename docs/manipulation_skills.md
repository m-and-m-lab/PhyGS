# Manipulation skills

Status: interface approved 2026-10-06. Two skills remain: `stow` and `look_at`. `open`/`close`
were removed on 2026-10-16, and `pick`, `place` and `move_along_axis` on 2026-10-18 (the
manipulation contract, [manipulation_contract/index.html](manipulation_contract/index.html)):
agents compose grasps, placements and drawer strokes from `move_to_pose`, end-effector
trajectories and the gripper commands, as the robot tests' routines do (`tests/robot_tests/routines/`).
This guide stays the source of truth for the skills; the contract is the source of truth for
the low-level commands.

## Goal

Agents use the robot in one of two ways (see the skill design notes in
[manipulation_planners.md](manipulation_planners.md#skill-design-notes-survey-2026-10-06)):

- **Orchestrating agents** (VLM or LLM orchestrators, as in HomeBody or Gemini Robotics-ER)
  call `look_at` and `stow`, plan moves with `move_to_pose`, run straight lines as
  end-effector trajectories and close or open the gripper, and replan from the feedback.
- **Trajectory agents** (VLAs such as MolmoAct, code-as-policies, or an agent with its
  own planner) send joint or end-effector trajectories and own their motion.

Both use the same `robot.manipulation` client, the same runtime and the same
independent evaluator. There is no second manipulation API.

## Physics rule

Grasped objects are never attached to the gripper in the simulator: no fixed joint, no
snapping, no "magic grasp". An object is held only by gripper contact and friction. If
it slips, hits something or loses grip, it drops, and `holding` becomes false.
"Attaching" a held object only adds its shape to the planners' collision checks (and the
planner's depth map); the simulator never sees it.

## Layers and collision responsibility

| Layer | Commands | Collision responsibility |
|---|---|---|
| Runtime | (not called by agents) | Always: joint, speed and acceleration limits; tracking-error stop; e-stop; exclusive joint ownership. No world collision avoidance. |
| Low-level commands | `move_joints`, `execute_trajectory` (joint or end-effector), `open_gripper`, `close_gripper`, `cancel` | None. The agent owns its trajectory, as with the real Spot SDK. |
| Planned move | `move_to_pose(pose, *, held=None, tool="hand")` | The environment's planner, around the obstacles in permitted observations, with `held` as part of the robot. |
| Plan only | `plan(*, goal=None, pose_xyzw=None, tool="hand", held=None)` | The same planner, to every arm joint's position (`goal`) or a tool pose; returns a Future of the `JointTrajectory` and moves nothing (previews). Execute it with `execute_trajectory`, or drop it. It is planned while the simulation runs: with the charged clock, `session.sleep` until it is done. |
| Skills | `stow`, `unstow`, `look_at` | The selected backend. `collision_avoidance_planner` avoids furniture and clutter in permitted observations; `direct` avoids nothing. |
| Evaluator | (not visible to agents) | Measures and records robot contact with everything except objects the task allows; it judges the task outcome. Nothing checks or stops a motion while it runs (the trajectory guard was removed on 2026-10-13). |

Within a planned move (`move_to_pose`, and the planned parts of `stow` and `look_at`):

- `held` (a `HeldObject`: box size, pose in the wrist frame, scene `name`) is the object the
  gripper carries, as the agent says. It is part of the robot for the plan, and the obstacle it
  names is left out (known boxes) or blanked from the planner's depth images. Nothing is
  inferred: without `held` the plan carries nothing.
- Unknown space is not an obstacle; the robot may move into space it has not observed,
  and the evaluator measures any contact. Observing first improves the plan.
- No silent fallback. A planning failure fails the command; it never switches to a straight
  move.

## Public API

Skill methods on `ManipulationClient` in `phygs/skills/manipulation/api.py` (the low-level
commands are in the contract):

```python
def stow(self, *, held: HeldObject | None = None) -> int:
    """Fold the arm into the transit posture navigation requires; a held object stays held."""

def unstow(self) -> int:
    """Unfold the arm from the stow to the pre-stow posture (nothing when not stowed); optional."""

def look_at(
    self, target: SkillTarget, *, distance_m: float = 0.5, elevation_deg: float = 45.0, held: HeldObject | None = None
) -> int:
    """Move the hand camera to one view of the target and aim it there, upright."""

@property
def episode_id(self) -> int:
    """Current episode; a trajectory for execute_trajectory must carry it."""

@property
def holding(self) -> bool:
    """Whether the gripper closed on an object and still holds it (from measured gripper motion)."""

def tool_pose(self, tool: str = "hand") -> tuple[float, ...]:
    """World pose [m, xyzw] of the tool (hand between the jaws, or wrist); where a line starts."""

def minimum_move_duration(self, positions: dict[str, float]) -> float:
    """Shortest move_joints duration [s] to positions [rad] within the motion limits."""

def bind_skills(self, skills) -> None:
    """Trusted setup only: bind the skill backend once."""
```

A skill returns a command id from the same sequence as low-level commands.
`feedback(id)` returns `SkillFeedback` for skills; `cancel(id)` stops the skill and its
current motion. While a skill runs, the agent's own manipulation commands are rejected.
Skills move the arm and gripper only; the base and legs stay locked.

Errors in the call itself raise immediately, like every other command: no bound
backend, a skill already running, a target type the task does not permit, or missing
localization. Outcomes in the world come back as feedback.

### Targets

`SkillTarget = PixelTarget | PointTarget | ObjectTarget`, defined in
`phygs/execution/manipulation/skills/types.py` and re-exported by the API.

| Type | Fields | Resolution | Information |
|---|---|---|---|
| `PixelTarget` (default) | `camera`, `stamp`, `x`, `y` [px] | Depth at that pixel in the **latest** capture. An older stamp or invalid depth fails with `target_invalid`. | Permitted observations |
| `PointTarget` | `position` [m], `frame` (`"world"` or `"body"`) | Used as given. | As permitted by the task |
| `ObjectTarget` | `object_id` | The object's box center. | **Privileged.** Rejected unless the environment grants object poses. Recorded in the report. |

### Feedback

```python
@dataclass(frozen=True)
class SkillFeedback(MotionFeedback):
    """Skill outcome from robot-sensed signals; ``reason`` is one sentence for the agent."""

    skill: str = ""
    failure: SkillFailure | None = None  # None when status == "succeeded"
    attempts: int = 0                    # 1 once the skill moved; skills do not retry
    holding: bool = False                # gripper is stopped on an object at the end
```

`SkillFailure` values reported by `stow` and `look_at`:

| Value | Meaning | Typical agent response |
|---|---|---|
| `target_invalid` | Old capture or no depth at the pixel | Observe again |
| `out_of_reach` | The view has no IK solution | Move the base closer |
| `path_unobserved` | The path passes unobserved space | Look at the area, retry |
| `planning_failed` | The planner found no path | Move the base or choose another view |
| `obstacle_changed` | New evidence blocks the moving arm | Re-observe, retry |
| `execution_stalled` | Tracking error or timeout | Re-observe, retry |
| `cancelled` | Cancelled by the agent or an episode reset | — |

The other values (`holding_object`, `not_holding`, `no_grasp`, `no_place`, `grasp_empty`,
`grasp_lost`) stay in the enum but no skill reports them since `pick` and `place` were removed;
`motion_failure(reason)` maps a low-level command's failure reason to a value, as the routines
do. The feedback contains only what a real robot can sense: joint and gripper positions,
tracking error, planner and map results from permitted observations. Ground-truth contact
forces and object poses go only to the evaluator.

## Composing grasps, placements and straight motions

The robot tests' routines are the reference compositions (`tests/robot_tests/routines/arm.py`,
numbers in `routines/params.yaml`); any agent may do the same or something else:

- **Straight approach** (`arm.approach_straight`): `move_to_pose` (planned, `tool="wrist"`,
  with `held` while carrying) to the pose a distance back from the end pose along an axis, then
  a straight `execute_trajectory(EndEffectorTrajectory(...))` from the measured tool pose
  (`tool_pose`) to the end pose. The straight part is not checked against obstacles: it may end
  in contact, as a grasp or a push does.
- **Straight line** (`arm.move_straight`): an end-effector line from `tool_pose()` along a
  world axis, keeping the orientation, at `line_speed_m_s` (0.1 m/s): lift, pull, back out.
  Execution refuses it before moving when it leaves the arm's reach or needs another arm
  configuration (`TrajectoryRefused`), and fails it while moving when the tool leaves its path by
  more than 2 cm or 0.05 rad. How far a pull got is the agent's to measure: a drawer may stop it.
- **Pick** (`arm.pick`): open; for each approach and grasp yaw, a straight approach to the
  grasp wrist pose (`grasp_wrist_pose`: the robot's `grasp` tool frame on the point where the
  approach line enters the object, its x along the approach; Spot's is 0.183 m along the wrist x,
  in the middle of the jaw gap at 0.029 m; an option refused before moving tries the next); close;
  record the held object (`arm.held_object`: the box in the measured wrist frame, resting on the
  `fixed_jaw` tool's pad, Spot's at -0.036 m, where the closing moving jaw pushes it, or centred
  when both jaws close); lift straight back out; succeed if `holding`. Returns the `HeldObject`.
- **Place** (`arm.place`): keep the wrist orientation (the object keeps its yaw), a straight
  approach down along the gripper axis to where the object's bottom is `place_clearance_m`
  (0.04 m) above the point (the planner keeps 1 cm clearance and a 2 cm map cell can hold the
  surface), open, straight back up, close the empty gripper. The object drops the rest; the
  place checks measured 1-3 mm error, upright. Nothing checks where it rests: the evaluator does.

Geometry limits found with the old skills still hold: Spot's fixed jaw sits 3.6 cm off the
gripper axis (so an object centred on the axis must be under ~7 cm wide; the jaw-gap middle
avoids that, gap 0.13 m with the moving jaw opened 75 deg), and a straight motion is
position-controlled (no force or impedance control): pulling along a line slightly off the
slide axis builds force (on the kitchen island the jaws wedged at ~320 N).

## Drawers

There is no drawer skill. The routines compose a stroke in `drawer.open_drawer` and
`drawer.close_drawer` (`tests/robot_tests/routines/drawer.py`): open the gripper; a straight approach into the part to the handle grasp
pose; close; a straight line along the slide by the stroke; open; move the fixed jaw off the bar
and back out with straight lines. `tests/robot_tests/episodes/check_drawer_stroke.py` checks it.

Handle grasp pose: the bar along the gripper's y (Spot: the jaw hinge), the approach into the
part, at the robot's `handle` tool frame: with a fixed jaw the bar rests on it (its axis a radius
above the frame along the closing axis), otherwise its axis is on the frame. Spot's `handle` is at
x 0.22 m, z -0.029 m in the wrist frame: just above the fixed jaw's tooth, so the bar enters
without pushing a free drawer away, and the closing jaw presses it down behind the tooth, which
hooks it for the pull (`gripper.tools` in `phygs/robots/spot/robot.yaml`).
To let go, open the gripper and move the fixed jaw off the bar before backing out: backing
out directly lets the tooth drag the drawer to its end stop.

Geometry limits (Spot gripper and arm, measured in the robot test):

- The closing moving jaw reaches 0.247 m along the wrist axis, so the drawer front must
  be at least ~3.5 cm behind the bar's axis (the Bear-house handle: 3.75 cm).
- From a fixed base the arm cannot pull a drawer the full 0.37 m: grasping a closed drawer
  far enough out stretches the arm straight, and the end of a long pull is over the body.
  The line is refused before moving (`TrajectoryRefused`, `out_of_reach`); the agent moves the
  base or pulls less.
- A level front grasp cannot reach a handle 0.155 m above the floor (the house
  nightstand's bottom drawer) from the standing robot at any base distance; the top
  drawer's (0.465 m) is reached with the drawer front 1.2–1.4 m ahead of the settled base
  (library IK, 0.2 m pull).
- Squeezing a handle fixed to the world makes the jaw vibrate in place; the controller
  measures stillness over 0.05 s (`MotionConfig.motion_window_seconds`).
- The jaws on a drawer bar hold the wrist roll a few hundredths of a radian off its target
  (0.039 rad at 250 Hz physics): trajectories end when the arm stops, not at the last sample.

## Stow and look_at

`stow` folds the arm into the transit posture (`stow_posture` in the skill config:
0, −3.13, 3.13, 0, 0, 0 rad). Navigation refuses to move unless every arm joint is within
0.18 rad of it, and the transit footprint was measured in it; a unit test keeps the three
equal. The skill closes the empty gripper, then moves the arm. `direct` makes a straight
joint move. The planner backend plans a collision-free path to the pre-stow posture
(`prestow_posture`: 0.3 rad less folded in the shoulder and elbow), then moves straight into
the stow. Sphere models of the robot call the folded stow itself a self-collision: in every
cuRobo sphere fit the forearm overlaps the shoulder by 2-3 cm (the meshes are clear), while
the pre-stow keeps at least 1.2 cm. Planned motions that start from the stow first move
straight back to the pre-stow, because planners reject the stow as a start: the planned skills
do it as a step, and the arm planner does it for `move_to_pose` and `plan` (the request starts
at the pre-stow and the trajectory begins with the straight move, since 2026-10-26). `unstow`
(Spot's "ready" after "stow") does only that move, for an agent that wants the arm ready. Both short moves
stay inside the robot's folded volume above its body. `stow` closes an empty gripper;
a held object stays held and is carried in the stow (2026-10-09). The planned part avoids
obstacles with `held`; the final straight move into the stow and the navigation footprint do
not account for it, so a large object can hit the body or furniture: the agent decides.

`look_at(target, distance_m=0.5, elevation_deg=45.0, held=None)` moves the hand camera
(`look_camera`) to one view of a point and aims it there: an object target's center, a
pixel's depth return or a point. The camera is `distance_m` from the target, on the
robot's side of it, `elevation_deg` above the horizontal, upright; the wrist pose follows
from the measured wrist-to-camera transform. Library IK must reach the view
(`out_of_reach`); the planner backend plans to it (`PosePlanner`, with `held`; refused:
`planning_failed`), `direct` moves straight. The skill succeeds when the arm has settled
at the view. It does not check what the camera sees and does not try other views: the
agent observes afterwards and chooses another view if it wants one. Example: from 1.4 m,
a 0.7 m view into the open drawer shows its front (depth 0.26 m); a 0.5 m view shows its
bottom.

## Step narration

Every skill says what it does now, one line per step, on the logger
`phygs.skills.<kind>`: `manipulation` (for example `look_at: point (world)
(1.25, 0.20, 0.47) m; view 0.60 m, 60 deg up`, then `look_at: planning and moving to the
view`), `navigation` (goal, route, replans, turning to the heading, each status change) and
`locomotion` (each new velocity command, `stop`). The current manipulation step is also
`SkillFeedback.step`. `RobotControl.create` attaches a collector (`control.narration`,
`phygs.recording.narration.Narration`) that prints each line with the episode
time, keeps the latest line per kind for video overlays (`lines()`), and keeps them all
(`log`). Checks may add their own lines on `phygs.skills.check`.

## Held objects in collision checking

A plan request with `held` (from `move_to_pose`, `stow` or `look_at`):

- carries the box (in the wrist frame, padded like obstacles) and the measured gripper angle.
  MoveIt attaches the box to the wrist with the gripper links as touch links and plans with
  the real gripper angle. cuRobo fills a reserved `held_object` wrist link (256 sphere slots)
  with spheres covering the box (cells of the planner padding, grown until they fit). It keeps
  one planner per gripper angle, because it fixes locked joints when the planner is built.
- leaves the obstacle `held.name` names out of the known boxes (else the task target); the
  depth map treats the box's current voxels as robot, like the arm's own voxels.
- Without `held`, nothing is carried in the plan. The robot UI, whose person does not say what
  the gripper holds, passes an assumed `held_object_size_m` box between the jaws while the
  gripper reports `holding`.

## Backends

| Backend | Moves | Purpose |
|---|---|---|
| `collision_avoidance_planner` | Plans to the view and to the pre-stow posture around every known obstacle, executed with `execute_trajectory` | Reference skills |
| `direct` | Joint-limited IK, straight joint moves; avoids nothing | Comparison with "go straight" agents and the real Spot |

Both issue only low-level commands, so ownership, expiry, limits, the tracking-error
stop and the e-stop always apply. The environment builds the backend with
`create_skills(...)` and binds it once with `bind_skills`. The planner backend needs a
joint planner and a pose planner from the environment (`JointPlanner`, `PosePlanner` in
`skills/types.py`, called with the goal and `held`), because the environment owns the planner
process and the map.

Configuration in the robot's `robot.yaml` (Spot's: `phygs/robots/spot/robot.yaml`;
the skill backend is `skills.backend` in `phygs/configs/environment.yaml`). The tool frames are the
gripper geometry the skills, the UI and the robot tests' routines read; every tool but `wrist` has
the gripper's axes (x the approach, z the jaws' closing axis). The routines' line speed and
clearances are in `tests/robot_tests/routines/params.yaml`:

```yaml
arm:
  stow: [0.0, -3.13, 3.13, 0.0, 0.0, 0.0]
  prestow: [0.0, -2.83, 2.83, 0.0, 0.0, 0.0]
  move_seconds: 2.0
gripper:
  tools:                                   # Poses in the wrist link [m]
    wrist: {position: [0.0, 0.0, 0.0]}
    hand: {position: [0.20, 0.0, 0.029]}   # The middle of the jaw gap
    grasp: {position: [0.183, 0.0, 0.029]} # Jaws cover an object's top ~5 cm
    handle: {position: [0.22, 0.0, -0.029]}
    fixed_jaw: {position: [0.20, 0.0, -0.036]}
camera_roles:
  look: [hand]
```

| Path | Content |
|---|---|
| `phygs/skills/manipulation/api.py` | Public skill methods, `SkillFeedback`, re-exported targets |
| `phygs/execution/manipulation/skills/types.py` | Targets, `SkillFailure`, `SkillError`, `SkillConfig`, `HeldObject` |
| `phygs/execution/manipulation/skills/runner.py` | Runs a skill sequence one low-level command at a time |
| `phygs/execution/manipulation/skills/backends.py` | `DirectSkills`, `CollisionAvoidancePlannerSkills`, `create_skills` |
| `tests/robot_tests/routines/` | Reference compositions: pick, place (`arm.py`), drawer strokes (`drawer.py`) |

## Example: orchestrating agent

```python
target = session.hints.target()  # Privileged; a perceiving agent finds it in its images.
held = arm.pick(session, target)  # routines.arm; or compose it: move_to_pose, a line in, close, a line out.
session.wait(session.manipulation.stow(held=held))
result = session.wait(session.manipulation.look_at(PointTarget(spot), held=held))
if result.failure == SkillFailure.OUT_OF_REACH:
    session.wait(session.navigation.navigate_to(x, y))  # move closer, then retry
```

## Compatibility

- `pick`, `place` and `move_along_axis` were removed on 2026-10-18 without a deprecation
  (project decision, contract section 11), with `ManipulationClient.held_object`; `open` and
  `close`, never released, were removed on 2026-10-16.
- `RobotControl.create(manipulation_pose_planner=...)` stays; deprecate it first, in a
  separate change, if a skill backend replaces it.

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Target types | All three; `PixelTarget` is the default; `ObjectTarget` only when granted. |
| 2 | Default backend | `collision_avoidance_planner`; `direct` for comparisons. |
| 3 | Self-collision guard in the runtime (as on the real Spot) | Not planned: no guard checks or stops a running motion (2026-10-13). |
| 4 | Out of scope now | Automatic view selection. (cuRobo is supported again; see manipulation_planners.md.) |
| 5 | Drawer axis for `open` | Known axis from the task at first (privileged, recorded). Removed with `open` (2026-10-16). |
| 6 | `place` relations | `"on"` first; `"in"` with drawers. |
| 7 | Internal retries | None (2026-10-09): retrying is the agent's decision. |
| 8 | Contact with the part being opened, outside the grasp | Tolerated (2026-10-12): the planners could touch the handle region during the whole grasp plan. Superseded 2026-10-13: the move to the start pose is planned around everything, the handle included; only the straight line may end in contact. |
| 9 | Guards and in-skill heuristics | Removed (2026-10-13): no trajectory guard and no "hand off the line" stop. Contact and base movement are recorded by the evaluator, not limited; the task outcome is judged. The arm controller's own limits (tracking error, timeouts) stay. |
| 10 | Straight motions | `move_along_axis` (2026-10-13): one skill for an approach (with an end pose) and a reach back (from the current pose), refused before moving when the start pose or the line is out of reach or would change the arm configuration. The straight part is position-controlled and not checked against obstacles. `open`/`close` removed 2026-10-16; `move_along_axis`, `pick` and `place` removed 2026-10-18 (row 11). |
| 11 | Fixed recipes | Removed (2026-10-18): `pick`, `place` and `move_along_axis` were recipes an agent should compose; the manipulation contract replaces them with `move_to_pose` (plan and move in one call, `held` explicit) and end-effector trajectories. `stow` and `look_at` stay skills. |

## Steps

History; `pick`, `place` and `move_along_axis` below were removed on 2026-10-18.

1. **Done:** `pick` with both backends. The 10-run grasp batch passes through
   `robot.manipulation.pick` (`output_check/skill_known_*`, `skill_strict_*`); 9 unit
   tests in `tests/test_manipulation_skills.py`. `direct` collides on the `near` layout.
2. **Done:** held-object shape in collision checking (not in physics). The robot test
   refuses a carry whose goal only the held object would hit, then carries the object
   over the clutter and back with 0 N object and arm contact: 10/10 runs
   (`output_check/carry3_known_*`, `carry5_strict_*`).
3. **Done:** `place`. The robot test sets the target on a different free table spot per
   layout: 10/10 runs within 1.4–6.2 mm, upright, 0 N furniture contact
   (`output_check/place_gate_*`).
4. **Done (2026-10-08):** `open` and `close` for drawers, and an approach direction for
   `pick`. Built in six sub-steps, each tested before the next: API (unit tests on the
   client), approach direction (top-down poses unchanged), the backends (handle grasp,
   straight pull, release), the part source (`scene/parts.py`), the robot test
   (`check_manipulation_planning.py --task drawer`, a Bear-house-style storage unit), and
   the gate. Gate (`output_check/gate4`): 24/24 runs, 6 layouts × known boxes / depth ×
   MoveIt / cuRobo. Openings within 6 mm of the request (0.20, 0.25 m, and to the limit),
   closed to 12–16 mm, base moved ≤ 2 mm, 0 N furniture contact; the blocked drawer
   reports `execution_stalled`, lets go and moves the base ≤ 11 mm. Limits found on the
   way are listed under [Drawers](#drawers) (jaw reach, arm reach from a fixed
   base, contact chatter). The old AO-Grasp drawer smoke tests and the AO-Grasp/SDK stack
   were removed. Removed 2026-10-16: `open`, `close` and `--task drawer`; drawers are
   composed by the agent and checked by `check_drawer_stroke.py` and the episode tests.
5. `look_at`, `stow` and a full episode.
   1. **Done (2026-10-09):** `stow` and `look_at`. Gate (`output_check/gate7`): 44/44 runs
      (6 drawer + 5 table layouts × known boxes / depth × MoveIt / cuRobo; one cuRobo run
      rerun after an RTX render hang, see manipulation_planners.md). `look_at` axis
      0.03–0.89 deg off the target, center depth within 1.6 cm; `stow` within 0.0022 rad
      of the transit posture, from every pick/place/open/close end state and after a
      blocked drawer; 0 N furniture contact.
   2. **Done (2026-10-12):** basic skills and the house drawer. Skills make one attempt
      (no internal retries) and `look_at` goes to the one view it is given; `stow` keeps a
      held object; a handle release backs out 6 cm and lifts 8 cm; every skill narrates its
      steps (below). MoveIt plans the arm with the guard's spheres. The drawer test uses the
      top drawer of the house `corner_nightstand`. Gate (`output_check/20261009-1729_gate10`
      and `20261009-2210_rerun10`): 42/44 (tables 20/20, drawers 22/24; open 0.201–0.373 m,
      base shift ≤ 7.9 mm, views ≤ 1.1 deg and 2.2 mm off, stow ≤ 2.2 mrad, 0 N furniture
      contact). Not passed: one MoveIt half-open close approach the guard refused on the depth
      map, and one cuRobo run that Isaac Sim crashed (segmentation fault) twice.
   3. **Done (2026-10-12), grid scene:** a full evaluated episode
      (`check_manipulation_planning.py --task episode`, known boxes): stow, navigate to the
      house nightstand, open its top drawer in two 0.15 m strokes with a step back between
      them, look inside, navigate to the table (a straight final leg), look, pick, stow
      holding, navigate to a place stance, look, turn the wrist top-down (`move_to_pose`),
      place into the drawer, close in two strokes, stow. Passed with MoveIt
      (`output_check/20261012-1616_episode_moveit`, `episode_sheet.png`) and cuRobo
      (`20261012-1633_episode_curobo`): stances 3.2-4.8 cm and <= 0.049 rad off, drawer 0.153 ->
      0.306 -> 0.161 -> 0.012 m, target inside after placing and after closing, 0 N furniture
      contact. Next: the Bear house (needs the storage revision's per-body solver iterations
      removed and the robot's foot material bound).

   Agent lessons from the episode (the skills stay basic; the agent plans around them):
   - One arm: open the drawer before fetching the object (`open` refuses while holding).
   - A fixed base pulls about 0.18 m with a 5 cm stance error: grasp at most ~1.13 m from the
     base, release back-out at least ~0.85 m. Open farther in strokes with a step back.
   - Planned low-level moves (`move_to_pose`) cannot start from the stow: it is a self-collision
     in the planners' sphere model. A skill (`look_at`) leaves it first.
   - `place` keeps the wrist's orientation; after a stow the object lies sideways, so turn the
     wrist top-down above the spot first. A top-down wrist reaches ~0.9 m, not 1.06 m.
   - The gripper needs ~30 cm between a drawer front and a cabinet top above it in the
     planners' sphere model (16 cm after one stroke was not enough).
