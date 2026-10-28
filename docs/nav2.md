# Nav2 navigation and measured geometry

The agent API stays `robot.navigation.navigate_to(x, y, heading=None, ...)`. Navigation runs on
ROS 2 Jazzy Nav2: Smac Lattice plans, MPPI (Omni) follows. The built-in A* planner and follower
were removed on 2026-10-25. Episodes and the robot UI build it with
`phygs.simulation.navigation.nav2_navigation` (footprint check, map source, servers;
see [navigation](navigation.md#public-api)); diagnostics can bind the factory directly:

```python
from phygs.execution.navigation.backends.nav2 import Nav2BackendFactory, Nav2Config

# Add this argument to the existing RobotControl.create(...) call:
navigation_backend_factory = Nav2BackendFactory(Nav2Config(namespace="spot_navigation"), maps=maps)
```

The adapter uses Nav2 `ComputePathToPose` and `FollowPath`; returned stamped body velocities
enter the existing locomotion client. Nav2 never steps physics or writes joints.

Nav2 runs beside the simulation, as beside a robot (since 2026-10-26). The simulation publishes the
clock, odometry, transforms and camera clouds and never waits for MPPI: each control period applies
Nav2's newest velocity command for what is left of `command_timeout` (0.2 s) after its stamp; without
a fresh one the last lease runs out and the runtime stops the robot, and a goal without one for
`command_grace` (2 s) fails. Route checks and displays read Nav2's newest published global costmap
(`global_costmap/costmap_raw`, 5 Hz). The simulation still waits for a route (`ComputePathToPose`,
once per goal and replan) and for cancellations. Before, it waited for each MPPI command and fetched
the full costmap every 0.1 s, so a slow Nav2 slowed the simulation. The local costmap (MPPI's) is a
6 m window that follows the robot; the global one covers the map.
Cancellation awaits external action termination, then requests a controlled stop.
At the first state publication after a runtime reset, the adapter waits for
active Nav2 servers and clears both costmaps before publishing camera observations.
This removes the previous episode without erasing observations gathered during
the new episode's initial scan. Starting a goal preserves those observations and
waits for current map/sensor input before planning. Camera clouds are published between goals
only while the arm is folded (an unfolded arm is not part of the transit footprint). Nav2
`NO_VALID_PATH` is reported as unreachable. `GOAL_OCCUPIED` is too, unless the goal is within
`occupied_goal_reach` (0.35 m) and asks for the fine approach's precision: then the fine approach
finishes it on the navigation map. A `FollowPath` failure with `NO_VALID_CONTROL` replans. Transform
errors, timeouts and invalid starting states remain failures.
An emergency stop terminates the episode. A transport timeout prevents connection
reuse: restart the external stack and bridge before another run.

One Nav2 stack per namespace and ROS domain. `Nav2Stack` starts `ros2 launch` with a parent-death
signal (`setpriv --pdeathsig TERM`), and the launcher starts each server the same way, so a run that
dies without closing (killed, crashed, timed out) leaves no servers behind. A stale stack answered
the next run's requests on the same domain ("parameters were not declared", commands from another
simulation clock; 2026-10-25). The bridge also refuses to start when it sees a second planner or
controller in its namespace. Nav2's log lines are prefixed `[setpriv-N]` since.

Map sources: with the privileged `known_map` grant, declared box geometry as a static map
(`occupancy_grid`); otherwise (the default) the robot's depth cameras as point clouds in Nav2's voxel
layer over a declared flat-floor prior (`pointcloud`); for exploration the prior is unexplored beyond
the spawn disk ([frontier exploration](#frontier-exploration)). All use permitted ground-truth
localization. None introduces SLAM, recovery behaviors or human correction. A future
robot-native implementation can implement
`NavigationBackendFactory` / `NavigationBackend` without changing agent calls.
Such a backend must honor reset, command ownership, terminal stops and the declared
world-frame goal contract; a different native map frame requires an explicit transform.

## Measured transit envelope

Navigation plans with Spot's **standing outline** with the arm folded (a fixed robot outline, as
standard navigation stacks use, rather than the swept legs):

Nav2's costmaps carry the outline as a rectangle `footprint` plus `footprint_padding` (0.05 m
plus any configured clearance), written by the launcher from the profile. Smac Lattice plans
position and 16 headings with it (the installed 5 cm "omni" primitives: forward, sideways and
turning moves, scaled for coarser maps), and MPPI's `CostCritic` scores its samples with it. The
fine approach checks its moves with the unpadded outline.

History: until 2026-10-22 both planners used the walking sweep's enclosing circle; until
2026-10-24 Nav2 used NavFn and Regulated Pure Pursuit with the circle enclosing the outline
(0.660 m). A rectangle with Smac Hybrid-A* and RPP failed on 2026-10-22 (RPP's ~0.13 m tracking
error brushed the dynamic obstacle; `output_check/20261022-1230_rectangle_nav2_padding`). The
lattice + MPPI configuration passed the Nav2 group 11/11 (`output_check/20261023-2100_nav2_mppi_v2`).

`phygs_robot_tests/components/check_footprint.py` measures it: the enabled collision shapes'
local bounds at actual articulated link poses, in a level frame at the base yaw, standing
after the reset and in the last 0.5 s of every stop (`standing` in the profile: bounds,
`padding` 0.05 m for the swinging feet and stopping drift, hull). It also records the swept
convex hull and enclosing circle of the walking motions (`envelope`, for information), and the
steady-state body velocity error and stopping distance.

The profile `phygs/robots/spot/navigation/transit.json` (schema 2, 250 Hz, 2026-10-22, 8 solver position
iterations, `output_check/20261022-1917_footprint_solver8`): standing outline x -0.60 to +0.61 m,
y -0.26 to +0.26 m (1.20 m x 0.52 m), padded 1.30 m x 0.62 m. The walking sweep (9,065 samples,
seven motions: forward, backward, both lateral directions, ±1.5 rad/s turns, combined) is
1.33 m x 0.81 m with a 0.758 m radius. Forward 1 m/s tracking RMSE is 0.026 m/s; stopping
distances are 0.40 m forward, 0.43 m backward and 0.75 m sideways (deceleration limit 1 m/s²).
See the profile for all directions, angular errors, actual end states and criteria.

This is a conservative envelope of **sampled local collision bounds**, not a
proof covering arbitrary postures or every combined velocity. Bounds can
exceed rounded mesh geometry. It excludes carried objects, extended-arm control,
uneven terrain, disturbances and untested speeds. Navigation validates root asset,
checkpoint, policy preprocessing and live collision-geometry hashes, posture,
timestep and sampled speed range. The diagnostic keeps the gripper posture fixed.
Recalibrate when these inputs change.

Grid rasterization can add up to half a cell diagonal. Nav2's
inflation layer grades traversal cost beyond its lethal footprint; the launched
configuration sets the inflation radius to robot radius plus three times the
selected decay distance. The known-map diagnostic uses 0.2 m decay (a 0.6 m
soft-cost horizon); observed-map diagnostics retain 0.3 m decay. This outer
radius is a cost horizon, not an additional forbidden zone. Measuring the robot removes the guessed 0.9 m radius; it does not
eliminate discretization or tracking margins.

The bridge derives yaw rate only when a new localization timestamp arrives.
Camera and control updates that read the same sample retain its yaw rate instead
of publishing a false zero. Episode reset clears that history.

## Inspect in IsaacLab

Run from `/workspace/isaaclab` (source ROS 2 Jazzy for navigation). Navigation is checked as an agent
uses it (`episodes/check_navigation.py`, rung 6 of the [robot-test ladder](../phygs_robot_tests/README.md)):
goals around a wall, to a heading and by velocity commands, goals refused in a wall, in a closed pen
or in furniture, and a cancelled walk, on the course (`nav_course`) and in the house (`nav_house`).
`--viz kit` opens Kit to watch; the footprint check has keyboard control with `--manual`:

```bash
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/episodes/check_navigation.py --viz kit --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_navigation
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_footprint.py --manual --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_footprint_manual
```

Automated checks (the default, headless) use the same scripts and YAML files:

```bash
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_footprint.py --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_footprint
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/episodes/check_navigation.py --spec scripts/PhyGS/phygs_robot_tests/configs/episodes/nav_house.yaml --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_navigation_house
```

The footprint check writes the transit profile to `DIR/transit.json`. To adopt a new
calibration, inspect `DIR/check.json`, then copy the profile into the robot's folder
(`navigation.footprint_profile` in its `robot.yaml`; Spot's is
`phygs/robots/spot/navigation/transit.json`). Do not overwrite it with a manual run; manual runs never
produce a passed calibration. Add `--viz kit` to watch automated motion.

## ROS 2 Jazzy setup

Navigation needs Ubuntu 24.04, ROS 2 Jazzy and IsaacLab's bundled Python 3.12. ROS stays
outside IsaacLab's own dependencies; only navigation imports it.

First configure the Ubuntu Universe and official ROS 2 apt repositories using the
[ROS 2 Jazzy installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
Its repository setup uses the `ros2-apt-source` package to install the ROS signing
keys and apt source. These packages must be available inside the environment/container
running IsaacLab, not only on the host. Then install the runtime and Nav2 servers:

```bash
sudo apt update
sudo apt install ros-jazzy-ros-base ros-jazzy-navigation2 ros-jazzy-nav2-bringup
# Optional visualization outside Kit:
sudo apt install ros-jazzy-rviz2
```

Frontier exploration (`navigation.explore()`) also needs explore_lite from
[m-explore-ros2](https://github.com/robo-friends/m-explore-ros2) on Nav2's costmap. It has no apt
package for Jazzy; build it once into an overlay (about 20 s):

```bash
sudo apt install python3-colcon-common-extensions
git clone https://github.com/robo-friends/m-explore-ros2.git /opt/explore_ws/src/m-explore-ros2
git -C /opt/explore_ws/src/m-explore-ros2 checkout 326cf8a0b487c34246bb8f3326afbcd69576dc60
source /opt/ros/jazzy/setup.bash
cd /opt/explore_ws && colcon build --packages-select explore_lite_msgs explore_lite \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

Then, in every shell that runs an episode or a navigation test, source
`/opt/explore_ws/install/setup.bash` instead of `/opt/ros/jazzy/setup.bash` (it sources Jazzy too)
and set `ROS_DOMAIN_ID` (one per person or simulator on the machine). Check that Isaac Lab's Python
loads the ROS bindings and finds explore_lite:

```bash
./isaaclab.sh -p -c "import rclpy, nav2_msgs, tf2_ros, explore_lite_msgs; print('ROS ok')"
```

The Nav2 install follows the [official binary installation instructions](https://docs.nav2.org/jazzy/getting_started/build_and_install/local_installation/).
The current experiment requires neither Gazebo nor SLAM. Use Jazzy consistently;
do not mix its message packages with another ROS distribution or install `rclpy`
from pip. The desktop ROS metapackage is an alternative if the full ROS GUI toolset
is wanted; `ros-base` plus Nav2 is sufficient for our bridge.

Verify imports using the actual IsaacLab interpreter after sourcing ROS:

```bash
cd /workspace/isaaclab
source /opt/ros/jazzy/setup.bash
./isaaclab.sh -p -c "import sys, rclpy, tf2_ros, rosgraph_msgs; from nav2_msgs.action import ComputePathToPose, FollowPath; from nav2_msgs.srv import GetCostmap; from geometry_msgs.msg import TwistStamped; print(sys.executable); print(rclpy.__file__); print('ROS/Nav2 imports passed')"
ros2 pkg prefix nav2_planner
ros2 pkg prefix nav2_controller
```

A successful system `ros2` command alone does not verify that bundled IsaacLab Python
can load the compiled ROS bindings. Resolve any import/shared-library failure before
launching the robot test. The existing Python 3.12 matches Jazzy's Ubuntu 24.04 target,
but the import check is still required. If using separate containers for servers,
configure DDS connectivity and use the same `ROS_DOMAIN_ID` in both.


The Nav2 bridge needs ROS Python bindings in the **same Python 3.12 interpreter**
used by IsaacLab. The Nav2 servers may be separate processes. Source an installed
ROS 2 Jazzy environment in both terminals; `rclpy`, `nav2_msgs`, `tf2_ros`,
`rosgraph_msgs`, `nav2_common`, navigation2 and nav2_bringup must be available.
Use a dedicated `ROS_DOMAIN_ID` and only one simulator/clock publisher per domain.

The episode harness starts the servers itself (`execution/navigation/backends/nav2_stack.py`) and
writes the experiment it launched them from to `nav2_experiment.yaml` in the run folder. To launch
them by hand from such a file:

```bash
cd /workspace/isaaclab
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=47
ros2 launch scripts/PhyGS/phygs/execution/navigation/launch/nav2.launch.py \
  experiment:=/absolute/path/to/nav2_experiment.yaml sensor_mode:=pointcloud
```

The launch file reads the experiment YAML and footprint profile (`experiment` is required;
`sensor_mode` is `occupancy_grid` for known maps, `pointcloud` for camera maps).
It configures Smac Lattice and MPPI (`phygs/configs/navigation/nav2.yaml`), two costmaps with the
measured footprint, matching speed limits, and simulation time. MPPI's `model_dt` is the
navigation period (0.04 s) and its horizon 2 s (`time_steps` = 2 s / period), so MPPI's model
step matches the command period. The adapter checks costmap geometry
before each goal and waits for both costmaps to acknowledge the current physical
obstacle cells before planning. It publishes occupancy without robot inflation;
Nav2 inflates this map once. TF is on `/tf` with namespace-prefixed frame names.
An identity map-to-odom transform is explicitly ground truth for this experiment.

MPPI tuning (2026-10-24, measured on a kinematic robot paced in real time and in
simulation): no `TwirlingCritic` (with it MPPI crept at under 0.1 m/s); Nav2's default sampling
accelerations (3, 3, 3.5) instead of the locomotion deceleration limits (MPPI drifted backwards);
`PathAngleCritic` mode 0 (face along the route); the lattice path is resampled densely before
`FollowPath`. Locomotion stands for commands at or below 0.025 m/s, so MPPI's small final
commands are raised to the lowest walking speed or end in the fine approach.

An absent ROS installation, missing server, mismatched geometry, stale command,
invalid frame or grossly excessive velocity fails explicitly. The adapter waits for
both Nav2 lifecycle nodes to be active before sending goals, re-sending its clock, transforms
and odometry on every poll for up to `bringup_timeout` (60 s): an activating costmap blocks its
node until it sees that transform, and a single TF message sent before the listener matched is
lost, so a state request the blocked node cannot answer yet is dropped and retried (the
`nav2_known_map` bringup race of 2026-10-22, one request timing out after 10 s). Nav2's own
lifecycle manager can also stall: it lost a configure response once and never went on
(`nav2_depth_map`, 2026-10-23). A node whose state has not changed for
`stalled_transition_seconds` (15 s) is driven through its next transition (configure, then
activate) by the adapter; a node already mid-transition rejects the duplicate. It caps controller
commands that exceed declared linear or angular speed by at most 50%, warning once
per goal; larger overshoots fail. The ROS-only timestamp epoch
advances across resets to fence previous episodes; simulation time still starts at
zero for each evaluator. Pausing physics while awaiting ROS never advances the
physical experiment clock.

Configuration references: [Jazzy Smac Lattice](https://docs.nav2.org/configuration/packages/smac/configuring-smac-lattice.html)
and [Jazzy MPPI](https://docs.nav2.org/configuration/packages/configuring-mppic.html).

See [optional pipelines](pipelines.md) for private service observations, factory-owned
dependencies and migration from the old environment map argument.

## Dedicated Spot/Nav2 pipeline test

`check_nav2_pipeline.py` starts the real ROS planner/controller servers and Spot
in one command for the house search (`check_house.py`, rung 13 of the robot-test ladder), and shuts
down the owned processes afterward. No separate terminal launching Nav2 is needed. Use an otherwise
unused ROS domain (default 47), and a fresh output directory. Navigation and obstacle avoidance
are agent tests: the episode harness starts Nav2 for them.

```bash
source /opt/explore_ws/install/setup.bash  # Jazzy plus explore_lite (the house search explores).
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_nav2_pipeline.py --viz kit --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_house_search
```

Runs are automatic and headless unless `--viz kit` is given.

Each output directory contains `nav2.log`, `simulation.log`,
`check.json` (process outcome and explicit pipeline ownership), and the simulation's result
`simulation/check.json` (measurements/assertions). The final line is
`<test>: <status>; <DIR>`. Interrupted runs are not passes. Missing servers or ROS bindings fail.

This is **real Nav2 planning and control**, not a complete native Nav2 perception
stack. All inputs use ground-truth localization. Depth unprojection is the bridge's; in `pointcloud`
mode (what episodes and the UI use) Nav2's voxel layer maps the camera clouds itself.
The launcher does not start AMCL, SLAM, BT Navigator or recovery behaviors.
Testing those requires a separate native sensor/TF adapter and stack configuration.

### Native PointCloud2 obstacle input

The `pointcloud` configuration unprojects Spot depth in the bridge, then sends calibrated
optical-frame `PointCloud2` with a capture-time transform to Nav2's `VoxelLayer` in both costmaps.
The only published occupancy grid is a declared planar floor within the configured bounds; it
contains no scene obstacle locations. In episodes and the UI it is unknown except a disk around the
robot's start (`point_cloud.explored_radius`, 1 m; since 2026-10-27): the robot does not know the
map. Nav2 plans through unknown cells (Smac `allow_unknown`), MPPI treats them as passable (the
costmaps track unknown space), the bridge's route check counts only cells of cost 253 and 254 as
blocked, and `navigation.explore()` goes after them. Without `explored_radius` (the diagnostic configs
other than the house search) the whole floor is declared free. Nav2 marks and clears obstacle cells from the sensor messages,
then inflates them. Camera access is declared for the private navigation service, while the agent
receives its selected observation view. Ground-truth localization and the measured transit envelope
remain explicit.

Obstacle avoidance on this costmap is rung 7 of the robot-test ladder (`episodes/check_dynamic_obstacle.py`,
spec `nav_dynamic`): a box moved into the planned route must reach the costmap and be walked around
or stopped for without contact, and a box put in view, then taken away, must be marked and cleared.

```bash
source /opt/explore_ws/install/setup.bash
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/episodes/check_dynamic_obstacle.py --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_dynamic_obstacle
```

Nav2 maps the cameras in a 3D voxel grid, the standard for depth cameras (since 2026-10-26;
`nav2.launch.py` `voxel_layer`). The grid spans the floor to the top of the height band in 16 voxels
per column (Nav2's limit; 7.5 cm at 1.2 m). Each point marks its voxel when it lies in the band
(0.1–1.2 m in episodes and the UI, 0.2–1.2 m in the diagnostic configs) and clears every voxel on its
ray from the camera, in 3D; a column with a marked voxel is an obstacle cell. The source keeps floor
returns (from -0.1 m) and returns above the band (walls, a ceiling, up to 10 m) for clearing only:
a ray to the floor or a wall where a box stood clears the box's voxels, and a ray over a box clears
only the voxels above it, so the box's column stays marked. A voxel no ray crosses keeps its mark:
returns above the band were dropped at first, and the top of a box taken away stayed
(`output_check/20261026-0030_voxel_dynamic`).

Why 3D: Nav2 keeps only the points within a source's height band, for clearing as well as marking
(Jazzy `ObservationBuffer::bufferCloud`). A 2D obstacle layer fed the clouds never cleared with floor
returns: a box that moved away stayed at 253 when only floor lay behind it
(`output_check/20261025-1023_clearing_without_fix`). A 2D floor-keeping clearing source cleared it
but its rays over a box wore away the box's edge cells, and the robot touched it in 2 of 2 runs
(`output_check/20261025-1027_erosion_with_a`, `..._with_b`). Laser scans from
`pointcloud_to_laserscan` worked (2026-10-25) but flattened everything to one height; the voxel layer
keeps the camera's 3D view.

The bridge sends one point per `point_cloud.voxel_size` (5 cm) cube, their centroid (a voxel grid
filter, as a real robot's pipeline runs before its costmap): the voxel layer raytraces every point it
receives. The layer has no time decay, SLAM or BT Navigator; something below the band or never in
view is not seen.

Cameras capture every 0.032 s (31.25 Hz). Episodes and the UI send each front camera's newest depth
to Nav2 every `point_cloud.period` (0.1 s, 10 Hz, as on a real robot), as the points of every
`point_cloud.pixel_stride`-th pixel row and column (2: a quarter of the points; at 6 m neighbouring
points are 2.8 cm apart, under the 5 cm cells), filtered on the GPU before the copy, then one point per
5 cm voxel. Full clouds at 31 Hz cost 25% of an episode's wall time, at 10 Hz 21%; with the stride 4%
(house, 2026-10-25). The diagnostic configs send every capture at full resolution, one point per 5 cm
voxel. Nav2 velocity command expiry and the runtime's physical stopping checks still apply.

Field of view decides what the costmap holds. The two front cameras see a box beside the route
only from the front, so its back stays unmarked; MPPI checks the footprint against lethal cells only,
and the robot's rear corner touched the box in 3 of 6 runs (`output_check/20261026-13*_dynamic_rep*`).
With Spot's five body cameras (`left`, `right`, `rear` are in `phygs/configs/environment.yaml`) the whole
box was marked and 6 of 6 runs kept 0.10–0.12 m (`output_check/20261027-12*_dyn5cam_b*`, video
`20261027-1211_camera_view_5cam`), at an episode real-time factor of 0.63–0.74 instead of 0.82–1.03.
Episodes still send only `frontleft` and `frontright` (`point_cloud.cameras`); the change waits on
moving the point-cloud filter to the GPU.

This adapter still invokes `ComputePathToPose` and `FollowPath` directly. For
camera-driven point-cloud navigation, it checks the remaining path against Nav2's
newest published global costmap every 0.1 simulation seconds and after a failed
controller action. A newly blocked route
cancels the old action, stops Spot, and requests a fresh Nav2 path. If Nav2
cannot find one, the goal ends as blocked after controlled stopping. This is a
limited route-invalidation check, not a BT Navigator or recovery behavior.

### Frontier exploration

`navigation.explore(timeout=None)` explores until no frontier is left, its budget ends or the agent
cancels it. It is one navigation command, audited like `navigate_to`; its feedback names the frontier
being approached (`Exploring: frontier 3 at (11.20, 7.45) m`), and it ends `succeeded` ("No frontier
left"), `timed_out`, `cancelled` or `failed`.

The frontiers come from explore_lite ([m-explore-ros2](https://github.com/robo-friends/m-explore-ros2),
built from source: README, Navigation installation), started by the bridge for each `explore` and
stopped when it ends. It reads Nav2's global costmap, picks the frontier (free cells next to unknown
ones) by distance and size every second of wall time, and sends it as a Nav2 `NavigateToPose` goal. A
BT Navigator would serve that action; the bridge serves it instead, so each goal becomes a leg of the
agent's command: followed like a `navigate_to` goal, at any heading, within
`explore_arrival_tolerance` (0.3 m). explore_lite hears each leg's outcome; a frontier given up
(unreachable, blocked) is not tried again, and a newer goal supersedes the leg being followed.

Exploration needs unknown space: `DeclaredFloorMap(explored=(x, y, radius))` declares the floor
unexplored except the validated spawn disk, and Nav2's voxel layer reveals free space as the cameras
see it. Routes may cross unexplored cells (Smac `allow_unknown`); maps without unknown space (the
flat floor, known boxes) are unaffected. The house search check runs it on the kitchen
(`check_nav2_pipeline.py`), then a coverage pass: exploration ends when
the floor is mapped, and rays from the 0.9 m camera over the island map the floor behind it, so a
low cube there is never seen; the check then walks to viewpoints spread over the costmap's free cells
and turns at each (`exploration.coverage`, house.yaml). Episodes and the UI start unexplored beyond
`point_cloud.explored_radius` of the start (since 2026-10-27), so `explore` works there too; with a
fully declared floor it refuses with "Exploration needs unexplored space". The settings are `Nav2Config.explore_*`;
explore_lite's output goes to `explore.log` in the run folder.

### Inspect the ROS side in RViz2

The diagnostic normally opens only Isaac Sim. Add `--rviz` to start RViz2 with
`phygs/configs/navigation/spot_nav2.rviz` on the same ROS domain:

```bash
source /opt/ros/jazzy/setup.bash
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_nav2_pipeline.py \
  --rviz --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_nav2_rviz_check
```

The view uses fixed frame `spot_navigation/map`. It displays `/spot_navigation/map`,
Nav2's global costmap, its published navigation footprint, TF frames,
the point cloud in `pointcloud` mode, and the path published by this adapter at `/spot_navigation/planned_path`. Toggle the
local costmap in RViz's Displays panel. Use `--ros_domain_id N` if domain 47 is
already occupied; the launcher gives RViz, Nav2 and Spot that same domain.
If starting RViz separately, use the same domain and config while the diagnostic
is still running:

```bash
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=47 rviz2 -d scripts/PhyGS/configs/navigation/spot_nav2.rviz \
  --ros-args -p use_sim_time:=true
```

The current bridge does not publish a ROS `robot_description` or joint states, so
RViz shows Spot's base and camera TF frames rather than a 3D Spot `RobotModel`.
Isaac Sim remains the view of the full robot and scene. RViz is an observer here;
its goal buttons do not call this adapter's `navigate_to` API. ROS topics disappear
when the one-command diagnostic finishes and shuts down its processes.
