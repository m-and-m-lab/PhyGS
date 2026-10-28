# Mapping

Navigation maps come from Nav2. The framework no longer builds its own occupancy map: the depth
mapper (`helpers/mapping.py`, `DepthMapper`), its navigation map (`ObservedDepthMap`), the frontier
selector (`helpers/exploration.py`, `FrontierExplorer`), the `check_mapping.py` diagnostic and the
`depth_map` input of `check_nav2_pipeline.py` were retired on 2026-10-26. Their last version is in
the git history (commit `81064fc`).

| Map | Who builds it | From | Used by |
| --- | --- | --- | --- |
| Nav2's costmaps (the default) | Nav2's voxel layer | the front cameras' point clouds, over a flat floor unknown beyond 1 m of the start (`point_cloud.explored_radius`) | episodes, the robot UI, the Nav2 checks |
| Nav2's costmaps (`known_map` grant) | Nav2's static layer | the scene's known boxes (privileged) | episodes that grant it, the navigation robot test on the course (`nav_course`) |
| Nav2's costmap of unexplored space | Nav2's voxel layer | the cameras, over a floor unexplored beyond the spawn disk | `navigation.explore()`, the house search check |
| Arm planning's depth map | MoveIt (Octomap) or cuRobo | the robot's own depth images, per plan | collision-aware manipulation ([manipulation planners](manipulation_planners.md)) |

How the voxel layer marks and clears, and why it is 3D: [nav2](nav2.md#native-pointcloud2-obstacle-input).
How exploration picks frontiers: [nav2](nav2.md#frontier-exploration).

## Check it

```bash
source /opt/explore_ws/install/setup.bash
# An obstacle moved into a planned route on the camera costmap, then a box put in view and taken
# away (its cells must be marked, then clear; the course is a room: the voxel layer clears only along
# rays that return).
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/episodes/check_dynamic_obstacle.py --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_dynamic_obstacle

# The house search: explore() over the unexplored kitchen, then a coverage pass, until a colour
# detector sees the cube.
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_nav2_pipeline.py --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_house_search
```

The search saves Nav2's costmap (`costmap.png`: grey unknown, white free, black obstacles, inflation
shaded) next to `check.json` in `<output>/simulation`; the obstacle-avoidance test's video shows it
in the NAVIGATION panel (`--record`).

An agent that wants its own map builds it from its camera stream (`session.stream()`, depth and the
capture-time transforms; see the [agent code chapter](advisor_review/agent_code.html)); Nav2 does not
read it.
