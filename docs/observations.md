# Observations

The shared Spot runtime now exposes `robot.observe()`. Observation reads do not
step physics or trigger captures. The runtime samples configured sensors after a
physics step and returns detached snapshots; modifying a returned tensor cannot
modify the simulator or subsequent observations.

## Configure once, read simply

Configure sensors in the scene YAML before scene construction. The
`phygs_robot_tests/configs/observations.yaml` example registers the existing mounted
front-left camera, a diagnostic pinhole camera mounted clear of the arm, and an IMU.
Existing camera assets use `spawn=None`; an optional `pinhole` section creates a
new camera. Width, height, prim paths, and requested data types are explicit.

After constructing and initializing the scene:

```python
from phygs.observations.types import CameraPermission, ObservationConfig
from phygs.robots.control import RobotControl
from phygs.runtime.episode import RunMode

robot = RobotControl.create(
    sim,
    scene,
    mode=RunMode.AUTONOMOUS,
    observations=ObservationConfig(
        joint_state=True,
        imu=True,
        localization="none",
        cameras=(CameraPermission("frontleft"),),
        state_period=0.02,
        camera_period=0.032,
    ),
)
robot.reset()
observation = robot.observe()
camera = observation.cameras["frontleft"]
points_optical = camera.points()
body_from_optical = observation.transform("body", camera.frame, camera.stamp)
points_body = points_optical @ body_from_optical[:3, :3].T + body_from_optical[:3, 3]
```

The environment runner continues calling `robot.step()`; an agent only consumes
observations and submits commands. Existing users that omit `observations` incur
no sensor capture work. Calling `observe()` without configuration fails explicitly.

## Coordinates and information sources

| Product | Convention and provenance |
| --- | --- |
| Joint state | Names accompany positions [rad] and velocities [rad/s], in articulation order; simulated encoders. |
| IMU | Body-aligned angular velocity [rad/s] and proper acceleration [m/s²], including +g upward at rest; ideal simulated IMU. |
| Body frame | +X forward, +Y left, +Z up; Spot root body. |
| Optical frame | +X right, +Y down, +Z forward; named `<camera>_optical`. |
| RGB | `uint8`, H×W×3. |
| Depth | Floating H×W values [m] along optical +Z, not Euclidean ray range. Pixels with no return, nonfinite or nearer than `depth_min` (0.2 m, the Gemini 2's minimum) are 0 with a false validity mask. The adapter masks on the renderer's device (the GPU) before one copy each of depth and mask to the scene device; masking a CPU copy cost more (verified bit-identical on 11,100 house frames, 2026-10-25). |
| Intrinsics | 3×3 pinhole matrix in pixel coordinates. |
| Extrinsics | 4×4 body-from-optical matrix at image capture time; ideal robot kinematics. |
| Localization | `none` by default. Explicit `ground_truth` supplies world-from-body and body linear velocity [m/s], marked privileged. |
| World frame | Authoritative simulator reference; +Z up. It is only exposed through permitted ground-truth localization. |

Quaternion inputs follow **(x, y, z, w)**. The existing robot YAML field remains
explicitly named `rotation_wxyz`; the scene factory converts it. Observation
transforms use matrices: `p_target = T_target_from_source @ p_source` in homogeneous
coordinates. Pose translations use metres; rotations are dimensionless.

No estimated odometry/localization provider is bundled. `localization="estimated"`
with a `LocalizationEstimator` is an extension interface; see
[pipelines](pipelines.md#localization-and-frames). No odometry
frame is fabricated from ground truth. Unknown localization modes fail.

## Timing, validity, and reset

Each packet has an episode ID, integer tick, and simulation time [s]. Each sensor
sample has its own capture stamp. Samples remain at their original time when
polled between captures; a packet can legitimately contain images older than its
joint measurements. Periods must be positive integer multiples of the physics dt.
The initial capture is at episode tick zero after fixed reset settling.
The default camera period is 0.032 s (31.25 Hz, every 8 ticks of the 0.004 s physics timestep).
The runtime renders at capture time to obtain fresh images. This is a simulation
time rate; wall-clock FPS depends on GPU load and simulation throughput.

The IMU is updated every physics tick, including settling, because its acceleration
calculation differentiates velocity at that interval. Its publication rate is the
configured state rate. Camera captures use the runtime camera rate. Missing
sensors, missing output products, or invalid calibration fail explicitly. Joint,
IMU, and localization samples carry validity flags; depth has per-pixel validity.

`packet.transform(target, source, stamp)` resolves only transforms present at that
exact measurement stamp. It supports inversion and composition, but does not
interpolate or extrapolate. Capture-time world-from-body is retained alongside
camera extrinsics only when localization permits it. Older packets remain usable
as independent snapshots; transform queries across episode identities fail.

Reset discards the live cache and starts a new episode. A failed reset cannot
return the previous episode's packet. Sensor capture exceptions terminate the
runtime as failed. Episode results record observation permissions, localization
mode, and capture periods.

## Permission boundary

Task/environment setup selects permitted joints, IMU, localization, cameras, and
RGB/depth products. Agents cannot request extra products through `observe()`.
Disabled state products are `None`; disabled cameras are absent. With localization
`none`, no world pose, body linear velocity, or world-frame transform is returned.
Camera extrinsics are local robot geometry, not localization.

This is an API information boundary for cooperating agents, not process isolation:
trusted setup and evaluators still have simulator access. Existing legacy SDK
image/state binders have not been migrated and are not covered by this contract.
The current observation adapter supports one Spot environment and ideal pinhole
RGB-D sensors. It adds no object tracks, detector, mapping, or navigation planner.

## Inspect and verify

From the IsaacLab root, inspect the live scene and RGB/depth panels:

```bash
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_observations.py --manual \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_observations_manual
```

This opens Kit, keeps Spot standing, and launches the red obstacle upward every
four simulated seconds. Observe the blue background appearing behind it. Camera
panels update at the configured camera rate. Close Kit to finish the manual run;
manual inspection is recorded separately from an automated pass.

Run the assertions headlessly (the default), or add `--viz kit` to watch:

```bash
./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_observations.py --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_observations
```

A camera-equipped runtime renders at its scheduled capture ticks; it does not
also render on the independent physics cadence. The trusted pipeline shares
held capture tensors internally across physics ticks, while each agent-facing
read remains detached. Camera frequency is measured in simulation time; actual
wall-clock delivery depends on physics, actuator processing, and RGB-D capture
throughput.

Headless runs with cameras (no GUI, no visualizer) render only at capture ticks, so
PhysX writes body poses to Fabric (which only rendering reads) only in the two physics
steps before each capture, and from restore through settling
(`simulation.fabric.FabricTransformWrites`). Physics is unchanged. Any extra
`sim.render()` between captures shows the poses of the last capture. Two steps,
because a prim under a body (a mounted camera) follows the body's write one step
late: with one step, the camera pose was one capture old while Spot walked.

Observation cameras are scene entries of type `simulation.cameras.RenderCameraCfg`
(built from the `sensors.cameras` configuration by `robots.spot.render_camera_cfg`),
not IsaacLab camera sensors. `simulation.cameras.RenderCameras` renders cameras with the
same resolution and data types in one tiled render product
(`rep.create.render_product_tiled`, as IsaacLab's tiled camera does) with `rgb` and
`distance_to_image_plane` annotators that deliver on the GPU. Each render cuts the tiles
into per-camera images once on the GPU (`reshape_tiled_image`), then each read copies one
image to the scene device. Six 640×480 cameras take 13.6 ms per frame (house, 2026-10-27).
Tiled and per-camera images match: at most one edge pixel per camera differs by more than
1 cm in depth. `RenderCameras(..., tiled=False)` keeps one render product per camera. IsaacLab's camera
sensor converted every image with CPU kernels when physics ran on the CPU, about 8 ms
per camera per frame. The camera pose is its parent body's physics pose times the fixed
mount read from USD (no Fabric read); intrinsics are computed as IsaacLab did (square pixels, centered principal
point). Without a GUI or visualizer, Kit's default viewport is not rendered.

Camera rendering requires RTX support even with CPU physics. The launcher enables
cameras itself. It writes `DIR/check.json`, RGB PNG files, depth
preview PNG files, and metric depth `.npy` arrays. Preview brightness spans 0–5 m;
this display conversion does not change the metric observations.

The automated diagnostic checks valid robot/sensor data, capture timestamps,
known-surface depth and frame alignment, optical point reconstruction, physical
occlusion changes, permission exclusion, immutable snapshots, and reset isolation.
It intentionally uses evaluator access to the obstacle position when computing the
surface error; that object state is never inserted into the public packet.

```bash
./isaaclab.sh -p -m pytest scripts/PhyGS/tests/test_observations.py -q
```
