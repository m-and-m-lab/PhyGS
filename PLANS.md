# Plans

What the framework does next. History is in [CHANGELOG.md](CHANGELOG.md); the boundaries these
items keep are in [ARCHITECTURE.md](ARCHITECTURE.md#boundaries).

## Next, when a user needs it

| Item | When |
| --- | --- |
| Object placement and drawer openings in the spec, and placing them again between episodes in one process | The benchmark's episode generator |
| Scene requirements and their checker (what a scene must name and contain) | The benchmark's episode generator |
| The start check split: the framework checks that the environment works; the episode declares its initial state (drawer openings) and the benchmark checks it | With drawer openings in the spec |
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
- navigate_to can accept a stance before the cameras have mapped the obstacle beside it; once the
  robot stands there with its footprint against it, Nav2 finds no control to walk away ("Observed
  route remained blocked"). Seen once in drawer_place (2026-11-01, the pick stance turned 30 deg
  by a table corner; 3 of 4 runs refused that stance and passed).
