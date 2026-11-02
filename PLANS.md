# Plans

What the framework does next. History is in [CHANGELOG.md](CHANGELOG.md); the boundaries these
items keep are in [ARCHITECTURE.md](ARCHITECTURE.md#boundaries).

## Next, when a user needs it

| Item | When |
| --- | --- |
| Placing objects and drawers again between episodes in one process, with another target and robot start (the known map, the arm planner's target and Nav2 follow); one process per episode until then | The benchmark's episode generator, if a process per episode is too slow |
| Scene requirements and their checker (what a scene must name and contain) | The benchmark's episode generator |
| A second robot's measured `phygs.composite.Tuning` (grip range, reach, option lists) | When a second robot needs the drawer and retrieve rungs |
| Framework fixes found in model runs (2026-10-30/31): a `PixelTarget` resolved once to a world point (not only from the latest capture); depth at an object's edge from the nearest surface around the pixel; the gripper still moving at the stop check and its tracking error | Before the next model runs that hit them |
| Estimated localization (the agent does not get its true pose) | A benchmark's onboard track |
| Base and arm moving together; a force-limited pull | Doors and fridges |
| An agent in its own process (a client over a wire); `fork` of the world state for dry runs | Agents that write code; agents run by others |
| The real Spot behind the same clients | Real-robot runs |

## Known gaps

- Differential-drive bases are not supported by the navigation stack yet.
- The five body cameras are parked until the voxel filter runs on the GPU.
- The robot test for obstacle avoidance has been flaky; it passed in every run on 2026-10-30.
- The straight lines of the composite skills (the last 0.19 m into a grasp, the lift, a drawer's
  stroke) run unchecked (`execute_trajectory`: no planner, no collision check); only the move to
  their start is planned. In the house drawer Spot's elbow clears the drawer above by little on
  the tilted grasp: a grasp 2 cm too deep (a box measured short) hit it in 3 of 4 pointing-oracle
  failures, and with the box measured right it still hit once in 4, from a stance 1.8 deg off
  square (the oracle stands square: 10 of 10, 2026-11-01). A check needs a configuration query in
  both planners (cuRobo, MoveIt) with the jaws and the grasped box left out.
- The cuRobo adapter reports every failed plan as `no_plan_within_budget`, also when cuRobo found
  the start or the goal invalid ("start or end state in collision"): the feedback hides the cause
  (a wrist at its limit hid that way on 2026-11-01).
- Taking the handle of a drawer that is already open: the wrist touches the front on the approach
  and, with no end stop behind it, pushes the drawer 4 cm closed, so the jaws close on nothing
  (the oracle on a spec that opens the grid drawer 0.25 m, 2026-11-01). Closed drawers sit at their
  stop, so the touch moves nothing.
- navigate_to can accept a stance before the cameras have mapped the obstacle beside it; once the
  robot stands there with its footprint against it, Nav2 finds no control to walk away ("Observed
  route remained blocked"). Seen once in drawer_place (2026-11-01, the pick stance turned 30 deg
  by a table corner; 3 of 4 runs refused that stance and passed).
