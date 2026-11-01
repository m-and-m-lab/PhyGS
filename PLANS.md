# Plans

What the framework does next. History is in [CHANGELOG.md](CHANGELOG.md); the boundaries these
items keep are in [ARCHITECTURE.md](ARCHITECTURE.md#boundaries).

## Next, when a user needs it

| Item | When |
| --- | --- |
| Object placement and drawer openings in the spec, and placing them again between episodes in one process | The benchmark's episode generator |
| Scene requirements and their checker (what a scene must name and contain) | The benchmark's episode generator |
| The start check split: the framework checks that the environment works; the episode declares its initial state (drawer openings) and the benchmark checks it | With drawer openings in the spec |
| A public layer of composed skills (pick, place, open and close) | Once it is settled what a skill should include; until then composed behaviour is agent-side code over `phygs.session`, as the robot tests' routines are |
| Stances computed from a robot's reach, for a second robot | When a second robot needs the drawer and retrieve rungs |
| Estimated localization (the agent does not get its true pose) | A benchmark's onboard track |
| Base and arm moving together; a force-limited pull | Doors and fridges |
| An agent in its own process (a client over a wire); `fork` of the world state for dry runs | Agents that write code; agents run by others |
| The real Spot behind the same clients | Real-robot runs |

## Known gaps

- Differential-drive bases are not supported by the navigation stack yet.
- The five body cameras are parked until the voxel filter runs on the GPU.
- The robot test for obstacle avoidance has been flaky; it passed in every run on 2026-10-30.
