# Point navigation

Navigation runs on Nav2 (Jazzy): the Smac Lattice planner and the MPPI Omni controller,
started by the episode and the robot UI alike. Nav2 builds its own map: by default its obstacle
layer maps the robot's two front depth cameras on a flat floor that is unknown beyond 1 m of the
robot's start until the cameras see it; with the privileged
`known_map` grant it instead gets the scene's boxes as a static map. Localization is ground truth.
Every privileged source is recorded. The built-in A* planner and path follower were removed on
2026-10-25.

## Public API

`simulation/navigation.py` builds Nav2 for a simulated Spot: it checks the live collision
geometry against the measured footprint profile, chooses the map source and starts the servers.
See the [Nav2 guide](nav2.md) for the setup and measurements.

```python
from phygs.simulation.navigation import nav2_navigation

navigation = nav2_navigation(
    stage=sim.stage, scene=scene, robot_config=robot_config, robot_dir=robot_dir,
    area=scene_def.map,              # bounds from the USD colliders, resolution, clearance [m]
    dt=dt, output=output / "nav2", nav2={"namespace": "spot_navigation"},
    point_cloud={"cameras": ["frontleft", "frontright"], "height": [0.1, 1.2], "max_depth": 6.0,
                 "period": 0.1, "pixel_stride": 2, "voxel_size": 0.05, "explored_radius": 1.0},
    start=(0.0, 0.0),                # the robot's start [m]: the map is unknown beyond explored_radius
)  # or known_boxes=(BoxGeometry(...), ...) instead of point_cloud
robot = RobotControl.create(sim, scene, ..., **navigation.control_arguments)
robot.reset()
command_id = robot.navigation.navigate_to(x=3.8, y=0.0, heading=0.0)
...
navigation.stack.close()
```

The environment runner continues calling `robot.step()`. The agent polls
`robot.navigation.feedback(command_id)` or calls
`robot.navigation.cancel(command_id)`. Distances are metres and heading is radians
in the world frame. Omitting heading requests position-only arrival. Position and
heading tolerances can be supplied to `navigate_to` (default 0.04 m and 0.04 rad).
`robot.navigation.explore(timeout=None)` walks explore_lite's frontiers on Nav2's costmap as one
command, on a map prior with unexplored space ([nav2](nav2.md#frontier-exploration)).

## Components and boundaries

- `execution/navigation/map.py`: immutable grid snapshots and segment checks (fine moves, the
  known map and the mapping diagnostics).
- `phygs/skills/navigation/api.py`: stable agent API and backend selection.
- `execution/navigation/interfaces.py`: complete navigation backend contract.
- `execution/navigation/fine_approach.py`: single-axis moves that finish a goal (internal to `navigate_to`, not a skill).
- `execution/navigation/backends/nav2.py` and `ros2.py`: the Nav2 adapter and its isolated ROS transport;
  `nav2_stack.py` runs the servers.
- `simulation/navigation.py`: Nav2 for a simulated Spot (footprint check, map source, servers).
- `execution/navigation/footprint.py` and `simulation/footprint.py`: sampled collision envelopes.
- `execution/navigation/metrics.py`: independent motion measurements and report comparison.
- `simulation/maps.py`: the privileged known map from declared boxes. Registered
  movable boxes use their actual physics poses; changed occupancy increments the
  map version and triggers replanning. Fixed boxes use their declared geometry.
- `tasks/point_navigation.py`: task configuration and evaluation from physical
  measurements, independent of the navigation client's success report.

With `known_map`, the map comes from the declared boxes, not from arbitrary USD collision
geometry; a scene must declare them faithfully. Without it, Nav2 sees what the cameras see.

Nav2's controller limits are `max_speed` (1.0 m/s) and `max_yaw_speed` (1.5 rad/s), within the
sampled folded-arm transit calibration; `period` (0.04 s) is MPPI's model step. These are command
limits, not guaranteed measured velocities.

The walking policy keeps standing for small velocity commands. Locomotion handles this:
`LocomotionController.DeadBand` (values in `command_dead_band` of
`robots/spot/locomotion/locomotion.yaml`) sends a command at or below 0.025 m/s (0.04 rad/s turning) as
0, so the robot stands (0.04 m/s until 2026-10-23), and raises one above it, up to 0.08 m/s (0.08 rad/s), to 0.08, the lowest
speed that walks in every direction (measured 2026-10-22). Every velocity command goes through it, from navigation or from
`command_velocity`.

Fine approach. The policy was trained with small commands on one axis at a time. Near rest it
does not follow small commands that mix axes: with the former built-in follower's proportional approach, 4 cm from a
goal it received 0.063 m/s sideways and 0.183 rad/s turning and stood still for 12 s, and goal
tolerances below 0.05 m timed out (2026-10-16, `output_check/20261016-1938_nav_tolerance_005_004`).
So for a position tolerance below `fine_approach_radius` (0.10 m), route following stops once within
that radius. From each settled pose it makes one move on one axis: a heading error above
`fine_turn_first` (0.3 rad) is turned first; then forward/back or sideways along the body axis
with the larger error; a smaller heading error is turned once the position is within tolerance (a
5-6 cm move turned the base 0.07-0.14 rad, so turning first every time alternated turns and moves
up to the move limit in the house, 2026-10-16). A move covers
its error in `fine_move_seconds` (1 s) at no less than the lowest walking speed and no more than
`fine_max_speed` (0.3 m/s) or `fine_max_yaw_speed` (0.5 rad/s). At the lowest speed (0.08 m/s) a
translation lasts as long as the measured short moves say it takes to cover the error
(`lowest_speed_moves` in `robots/spot/locomotion/locomotion.yaml`, `FineApproach.Coverage`): from standing, 0.5 s
to the right covered 0.4 cm and 1 s 5.7 cm, so sized as error / speed the 3 cm house and grid
stances timed out after 25 right moves of 4-10 cm (2026-10-22). It ends with the runtime's
controlled stop (ramp + 1.6 s); the next pose is measured once settled. Each move is checked
against the map with the unpadded footprint at the base heading (`blocked` if it would touch an
obstacle). The moves continue until the goal is reached, the time budget ends (`timed_out`) or
the command is cancelled. Outside the fine approach (a position tolerance at or above the fine
radius), a settled pose outside the tolerances is corrected by following a new route in, at most
`max_goal_corrections` (3) times; then the command fails with the remaining errors in its reason. Before this bound (2026-10-23) the house search looped on an unreachable 0.04 rad
heading tolerance until its budget: Nav2's controller does not turn in place for small errors.
Stop drift and turning in place shift the base 1-2 cm, past the entry radius, so the fine
approach keeps the goal until a settled pose is beyond `fine_approach_exit_radius` (0.25 m); only
then does route following take over again. Open floor, 5 goals (1.5 m ahead, 1.2 m left facing
left, a turn back, 2.5 m diagonal, 0.3 m), 0.03-0.05 m tolerance: 5/5 at 250 Hz, median
8.3-11.5 s, at most 7 moves (`output_check/20261016-2029_fine_approach_hysteresis`). At 250 Hz the
standing base creeps 1-2 cm in the 2 s after success.

Nav2 plans Spot's measured standing outline with the arm folded (1.21 m x 0.58 m) as a
rectangle, plus a 0.05 m padding; see [nav2](nav2.md#measured-transit-envelope). Fine-approach
moves are checked against the map with the unpadded outline.

The arm must remain within the declared folded posture. Arm commands are rejected
while navigation is active; unsupported posture fails explicitly. A held object may be
carried in the stow; the envelope does not cover it. Stairs, uneven terrain and
extended-arm navigation are outside this initial scope.

The footprint profile is tied to the robot's collision geometry by `geometry_sha256`. The
link-frame collider bounds are rounded to 1 nm before hashing (2026-10-12): they are composed
through world transforms, and spawning away from the origin changed them by ~1e-16 m.

Invalid goals and disconnected routes return `invalid_goal` and `unreachable`.
A newly blocked route stops and returns `blocked`. Cancellation stops before
returning `cancelled`. Direct motion commands supersede navigation and produce
`overridden`; navigation does not overwrite the new command. Reset cancels old
commands. Emergency stop terminates the episode. Success requires a settled pose
within both goal tolerances; residual stopping drift is corrected within the
original time budget.

## Evaluation and inspection

Navigation is checked as an agent uses it: `phygs_robot_tests/episodes/check_navigation.py` (rung 6
of the [robot-test ladder](../phygs_robot_tests/README.md)) sends `navigate_to` goals through the
agent session and judges arrival with the evaluator below; `episodes/check_dynamic_obstacle.py`
(rung 7) moves a box into the route as a scripted test perturbation.

The evaluator checks physical position, optional heading, low linear/angular speed,
and an arrival hold interval. It records elapsed simulation time, XY travel distance,
falls and collisions. Navigation failure/cancellation aborts task evaluation with
the reason recorded. Skill success alone cannot establish task success.

The diagnostic evaluates every physics tick. A collision is a rising threshold
crossing of the net contact-force norm on a declared obstacle (default 1 N).
The fixtures avoid obstacle-to-obstacle and ground contacts, so these events measure
robot contact in this scene. This definition is not a general contact classifier
for arbitrary scenes. Falls use minimum base height and upright-axis thresholds.
No path-efficiency metric is reported.

The unobstructed task runs through both the navigation client and a direct velocity
agent with the same task, reset, clock and evaluator. Reports include the actual
settled initial state, configuration, policy/asset hashes, events and physical
samples. The manual launcher offers goal entry, cancellation and reset; green
markers show route vertices.
