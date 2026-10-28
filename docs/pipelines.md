# Agent inputs and optional navigation services

The environment does not impose a localization -> mapping -> planning pipeline.
An agent can own all of these, consume selected products, or call a complete
navigation service. Acting APIs remain in `phygs/skills/*/api.py`.

| Arrangement | Environment supplies | Agent owns |
| --- | --- | --- |
| Observations only | Images, calibration, configured joints/IMU; motion API | Localization, representations, planning/control |
| Localized observations | Observations plus a declared pose provider; motion API | Mapping and planning/control |
| Complete navigation | A configured `navigate_to` service and selected observations | High-level decisions and any private memory |
| Mixed processing | Explicitly registered, optionally exposed products | Remaining processing and decisions |

An agent's scene graph need not be a navigation costmap. Private model state needs
no framework registration. A registered processor can publish its own typed
snapshot; the pipeline never converts all products into occupancy grids.

## Environment assembly

`runtime/pipeline.py` owns capture, reset and processing scheduling. It separates:

- `PipelineConfig.capture`: actual sensor capture configuration and rates.
- `PipelineConfig.permissions`: task-wide permitted observations, including those
  used privately by services. Additional privileged sources are explicitly listed.
- `PipelineConfig.agent`: products returned by `robot.observe()`.
- `ServiceDeclaration`: each component's inputs, sources, dependencies and outputs.
- `ProcessingService`: an optional processor, with opt-in agent publication.

Task/environment code constructs these objects; agents continue using `observe`,
`product`, and skill commands. No processor or navigation stack is started merely
because cameras or localization are configured.

```python
from phygs.observations.types import CameraPermission, ObservationConfig
from phygs.runtime.pipeline import PipelineConfig

capture = ObservationConfig(
    cameras=(CameraPermission("inspection"),), localization="ground_truth",
)

# Agent supplies its own localization/mapping. This preset disables ground-truth capture.
pipeline = PipelineConfig.observations_only(capture)

# Alternatively: localization supplied, mapping still belongs to the agent.
pipeline = PipelineConfig.localized(capture)

robot = RobotControl.create(
    sim, scene, mode=RunMode.AUTONOMOUS,
    pipeline_config=pipeline, allowed_actions=("motion",),
)
```

`RobotControl` is trusted environment assembly; an agent should receive the observation
and action APIs, not the simulator or pipeline internals. These contracts enforce
configured information flows, not an adversarial sandbox for arbitrary Python code.

## Complete navigation with private service inputs

Navigation factories own their dependencies. `NavigationContext` provides runtime
command authority, a filtered observation view, and navigation settings. It has no
map argument. A stack can use ROS components, internal learned representations,
its own localization, or onboard navigation. A factory declares `configuration`
and `declaration` and returns a backend implementing the navigation lifecycle.

```python
from dataclasses import replace
from phygs.execution.navigation.backends.nav2 import Nav2BackendFactory, Nav2Config
from phygs.observations.types import ObservationAccess
from phygs.runtime.pipeline import PipelineConfig

available = ObservationAccess.from_config(capture)
pipeline = PipelineConfig(
    capture=capture,
    agent=replace(available, localization="none"),
    permissions=available,
    privileged_sources=("privileged_known_box_geometry",),  # only for this known-map experiment
)
factory = Nav2BackendFactory(Nav2Config(namespace="spot_navigation"), maps=maps)
robot = RobotControl.create(
    sim, scene, mode=RunMode.AUTONOMOUS, pipeline_config=pipeline,
    navigation_backend_factory=factory, allowed_actions=("navigation", "motion"),
)
```

Nav2 is the only stack (the built-in planner was removed on 2026-10-25). The grid-fed
configuration sends framework occupancy to Nav2's static layer. The `pointcloud`
configuration sends permitted camera depth as `PointCloud2` to Nav2's obstacle
layer, alongside a declared flat-floor prior. The factory's map input is an
adapter-specific dependency; the shared navigation interface does not require a
map. Real-robot adapters, sensor-based localization, and Nav2 BT orchestration
remain future work. There is no silent fallback between stacks.

An agent can receive only raw products while its navigation service uses permitted
localization or a privileged map. That is **assisted navigation**, not an
unprivileged experiment. Episode results record each service's access and transitive
information sources, plus the agent selection and exposed products. Forbidden
service inputs or privileged sources fail before robot initialization.

## Optional mapping

A processing service is any object with ``reset(episode_id)``, ``integrate(packet) -> bool`` (False:
nothing new) and ``snapshot()``. Navigation does not need one: Nav2 maps the cameras itself
([mapping](mapping.md)).

```python
from phygs.runtime.pipeline import ProcessingService, ServiceDeclaration

mapper = MyMapper()  # Your own: reset, integrate, snapshot.
service = ProcessingService(
    ServiceDeclaration(
        "mapping",
        ObservationAccess(
            localization="ground_truth",
            cameras=(CameraPermission("inspection", rgb=False),),
        ),
        outputs=("occupancy_grid",),
    ),
    mapper,
    expose_to_agent=True,
)
# Pass processing_services=(service,) to RobotControl.create(...).
# The runtime resets and updates it; the agent may read robot.product("mapping").
```

A stack using this processor's result declares `dependencies=("mapping",)` on its
factory. Components are registered in dependency order; missing dependencies and
duplicate names fail explicitly. Reads never update a processor or step physics.
Repeated camera captures are the processor's to deduplicate. Processor exceptions end
the episode through the same runtime failure path as sensor/controller errors.

## Localization and frames

`LocalizationEstimator` accepts only its declared observation inputs. Implement
`reset`, `update(packet)` and `localize(stamp)`, then supply it as
`localization_estimator=`. Capture must exclude ground-truth localization;
permissions and consumers explicitly select `localization="estimated"`.

The estimator declares a fixed reference frame and must supply poses at the actual
camera capture times, including retained images from slower cameras. No latest-pose
substitution or world-state fallback is performed. The implementation validates
mode, validity, frame, timestamp, episode, tensor shape and finiteness. This is an
extension interface, not a bundled or physically validated localization algorithm.

`NavigationConfig(reference_frame="local_map")` and a corresponding map frame let
the grid-backed stacks consume that provider. Map, localization and goal frames
must agree. Configure `localization_mode="estimated"` on the selected factory.

`robot.navigation.navigate_to(x, y, heading, frame="body")` resolves a relative goal
once, at submission, using the navigation service's permitted pose. Omitting `frame`
uses the stack's declared fixed frame. An unsupported private scene-graph frame
fails explicitly; coordinates are never silently relabelled as world coordinates.
A model that handles relative goals internally can wrap this in its own backend;
the public body's conversion currently requires a supplied localization view.

Agent filtering reconstructs transform graphs from selected sensor-to-body
extrinsics and, only when permitted, localization edges. Hiding the pose field also
hides its world/map transforms. Calibration and kinematics remain available with
selected cameras without exposing world localization.

## Execution and migration

Direct locomotion while navigation is active is rejected. Cancel navigation, let
the environment run until its controlled stop completes, then submit direct motion.
Public navigation feedback omits the planned path because it can reveal private
localization and map geometry. The environment runner uses private diagnostics for
visualization and measurement. Navigation's internal motion commands still pass
through runtime ownership/expiry.
Emergency termination cannot resume the same episode.

Environment constructors no longer take `map_provider=`. Supply a factory that owns
its map, if needed. Custom factories now implement `factory(context)` rather than
`factory(runtime, locomotion, maps, config)`. The agent's `navigate_to` call remains
the same. Point-navigation evaluation uses physical measurements independently of
whether the agent receives localization.

## Inspect the configurations

Run from `/workspace/isaaclab`; add `--viz kit` to watch, and `--manual` for the
existing manual inspection modes. There are no duplicate scripts or YAML files.

```bash
# Observations and direct-motion interface, without supplied localization or mapping.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_observations.py \
  --preset observations --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_pipeline_observations

# Observations with ground-truth localization, without a mapper/navigation stack.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_observations.py \
  --preset localized --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_pipeline_localized

# Runtime-managed navigation on Nav2's camera costmap, driven through the agent session.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_dynamic_obstacle.py --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_pipeline_nav2

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest \
  scripts/PhyGS/tests/test_pipeline.py -q
```

The obstacle-avoidance test reads ground truth and Nav2's costmap on the test side
(`Environment.truth()`, the navigation backend); neither reaches the agent.

The [dedicated Nav2 robot test](nav2.md#dedicated-spotnav2-pipeline-test) launches
real planner/controller servers alongside Spot for the house search, on camera point-cloud
input. It records the ownership boundary explicitly: `pointcloud` feeds Nav2's voxel layer from
depth. It includes no SLAM.

### Physics device and command throughput

Physics device selection remains explicit through `--device cpu` or
`--device cuda:0` on the robot test launchers. CPU is the current diagnostic
launcher default, not a framework requirement or an automatic fallback. The
single-robot measurement does not establish the best device for a larger scene;
compare the same scene, sensors, timestep, and robot configuration on each device.
Static scene size alone is not a sufficient predictor: active bodies, contacts,
rendering load, and device synchronization also affect throughput.

The world adapter retains the last submitted joint targets and only writes changed
values. Reset invalidates this cache. All commands must continue through the
runtime-owned adapter; external writes to articulation targets bypass its contract.
Actuator processing and physics still run on every 0.004 s tick. Python velocity
commands are validated on the host and converted to device tensors only on policy
inference ticks; tensor inputs retain device-side validation. Ownership, expiry,
stopping, and the 0.02 s policy inference interval are unchanged.

## Frontier exploration

`navigation.explore()` is one navigation command: explore_lite picks frontiers on Nav2's costmap and
the backend walks to each ([nav2](nav2.md#frontier-exploration)). It sees only Nav2's costmap, never
simulator geometry or target locations. An agent can instead choose its own viewpoints from its
camera stream and call `navigate_to`, or command locomotion directly.

`tests/robot_tests/components/check_house.py` (through `check_nav2_pipeline.py`) runs it with ground-truth localization
on the kitchen (`exploration.bounds`), unexplored beyond the validated spawn disk, while a colour
detector watches the camera; when exploration has mapped the floor without seeing the cube, a
coverage pass walks to viewpoints spread over the costmap's free cells and turns at each. It ends
when the cube is seen. This is a bounded diagnostic, not a
claim of reliable whole-house search. Camera frequency uses simulation time.

Nav2 (MPPI) follows the route; within `fine_approach_radius` (0.10 m) the backend stops Nav2 and
finishes with single-axis moves (`execution/navigation/fine_approach.py`). With the deployed policy, the PhysX-drive stand and 8 solver iterations the
house navigation check passes at 0.04 m / 0.04 rad (2026-10-22,
`output_check/20261022-1738_solver8_all`).
Do not treat the reference search helper as a guarantee of complete whole-house
visual coverage or target discovery.
