# Runtime-backed Spot control

The environment binds Spot once. Agents receive small clients; they do not load
checkpoints, acquire joint resources, adjust actuator gains, or tick the policy.

```python
from phygs.robots.control import RobotControl
from phygs.runtime.episode import RunMode

# Environment setup: sim and scene have already been initialized.
control = RobotControl.create(sim, scene, mode=RunMode.AUTONOMOUS)
control.reset()

# Agent command: body x/y velocity [m/s], yaw velocity [rad/s], duration [s].
command_id = control.locomotion.command_velocity(0.4, 0.0, 0.0, duration=2.0)

# The environment's existing loop calls control.step() once per physics tick.
# Agents may query feedback and request stopping later:
feedback = control.locomotion.feedback(command_id)
stop_id = control.locomotion.stop()
```

An independent arm command uses the same runtime:

```python
arm_id = control.arm.command_joint_positions({"arm_sh0": 0.3}, duration=1.0)
arm_feedback = control.arm.feedback(arm_id)
```

Arm commands interpolate from measured positions and hold unspecified arm joints.
They validate the configured joint limits. Completion means the interpolation or
stop sequence has finished; task evaluation must independently measure physical
success. The gripper holds its initialized target; a gripper client is not part
of this milestone.

## Command behavior

- The default velocity-command lifetime is 0.5 s. Durations must be positive,
  finite integer multiples of the physics timestep; incompatible values fail.
- Updating velocity renews ownership without resetting policy history.
- A replaced command reports `overridden`; feedback also reports `active`,
  `stopping`, `done`, `cancelled`, `interrupted`, or `failed`.
- Expiry and `stop()` both request zero velocity. The policy command slows to zero at
  most at 2.0 m/s² forward/back, 1.0 m/s² sideways and 2.0 rad/s² turning
  (`command_deceleration` in `robots/spot/locomotion/locomotion.yaml`; speeding up is not limited), and the
  legs stay owned for that ramp plus 1.6 s (`stop_stand_seconds`). The policy hands over to
  standing once the base is nearly still, turns slowly, no foot is in swing and all feet are
  down; or from 0.7 s once the feet are down; or at 1.5 s at the latest (`stand_switch`).
  Standing, PhysX joint drives hold the legs (implicit PD, 2000 / 80 hips, 2500 / 80 knees,
  `stand_drive`; no actuator delay) where they are, not at the default posture. With every leg on a
  PhysX drive, TGS needs 8 position iterations (Spot's `robot.yaml` `articulation`, was 4) and gravity
  applied in every iteration (`robots.scene.simulation_cfg`): with 4 and once per step the arm sat 6-8 mrad off target and
  joints reported 0.04 rad/s while still (2026-10-22). Ownership then releases; the legs keep the standing targets
  (`LocomotionController.hold_targets`), or measured positions if the policy did not stand.
  Holding the measured, load-deflected positions dropped the stance preload: the base sank by
  about load / stiffness after each stop (median 1.7 cm at 250 Hz) and tipped
  after some left turns at 250 Hz (2026-10-16, `output_check/20261016-2119_turn_tip_check`).
- Reset restores physical state, gain buffers, and policy memory. Desired PD
  velocity and effort are zero; restored physical velocities are separate state.
- Emergency stop terminates the episode through `control.runtime.emergency_stop`.
  `control.reset()` creates a new episode and preserves old command feedback.
- Set `RunMode.HUMAN` only for human-operated experiments. The migrated keyboard
  launcher uses that mode; autonomous controller registration rejects human input.

Goal navigation uses `robot.navigation.navigate_to(...)`; see [navigation](navigation.md).

## Preserved policy contract

The checkpoint and training configuration live in `phygs/robots/spot/locomotion/`
(`policy.pt`, `env.yaml`, `locomotion.yaml`; the wrapper is `robots/spot/policy.py`). Agents use `LocomotionClient` from
`phygs.skills.locomotion.api`, normally obtained as `robot.locomotion`.
The internal policy computes targets; only the runtime applies them to the robot.
There is no standalone policy-writing API or compatibility wrapper.

| Property | Contract |
| --- | --- |
| Command | Body-frame `(v_x, v_y, yaw_rate)`, units `[m/s, m/s, rad/s]` |
| Physics / target application | 0.004 s (250 Hz, `physics_rate_hz` in `robots/spot/locomotion/locomotion.yaml`); trained at 0.002 s |
| Inference | Every 5 ticks: 0.02 s (50 Hz) |
| Observation | 60 values: body linear velocity (3), angular velocity (3), projected gravity (3), command (3), default-relative leg position (12), leg velocity (12), previous action (12), arm position relative to the folded training default (6), arm velocity (6) |
| Action | 12 joint-position offsets, scaled by 0.2 and added to default leg positions |
| Ordering | Existing robot joint resolution with `preserve_order=False`; resolved names saved in the simulation report |
| Memory | Previous action, cached action, inference phase, zero-command timer; cleared on reset/acquisition |
| Actuator ownership | Leg position targets, explicit-PD gains while walking and PhysX drive gains while standing; arm and gripper are never written by locomotion (the arm is only read) |
| Actuator models | Hips delayed PD clipped at 45 N·m, knees delayed PD with the angle-dependent torque limit, delay 0–8 ms drawn per reset (0–2 ticks), as in training |
| Information source | Simulator-derived body velocity, gravity and joint state, explicitly recorded |

`robots.spot.spot_actuator_cfgs` builds the actuators in one of two layouts with
identical torques. `fused` (default, `robot.actuator_groups` in the robot configuration):
`robots.actuators.SpotLegActuator` for the twelve legs and one implicit group for the arm
and gripper (`spot_legs`, `arm_and_gripper`). `isaaclab`: IsaacLab's `DelayedPDActuator`,
`RemotizedPDActuator` and two implicit groups (`spot_hip`, `spot_knee`, `arm_hold`,
`gripper_hold`). IsaacLab's articulation spends most of an explicit actuator's time per
group, so the fused layout takes 0.61 ms per physics step instead of 1.19 ms (single
environment, CPU). Use `isaaclab` for IsaacLab features that need its classes, such as
Newton-native actuators. `tests/test_spot_actuators.py` requires equal torques for every
delay, several environments and partial resets.

The runtime rejects a scene timestep incompatible with the checkpoint. It does
not resample inference or silently change the house scene's configured timestep.
The checkpoint is packaged with its existing YAML files; no model download occurs.

## Physical checks

From the IsaacLab workspace root:

```bash
./isaaclab.sh -p -m pytest \
  scripts/PhyGS/tests/test_episode_runtime.py \
  scripts/PhyGS/tests/test_locomotion_runtime.py \
  scripts/PhyGS/tests/test_spot_locomotion_api.py \
  scripts/PhyGS/tests/test_locomotion_launcher.py -q

./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_locomotion.py \
  --device cpu --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_locomotion
```

Add `--viz kit` to display the automated diagnostic on a machine with a GUI. With
`--manual`, the same launcher opens Kit for keyboard control:

```bash
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_locomotion.py --manual \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_locomotion_manual
```

The automated check uses real physics and the real checkpoint. Its YAML declares
commands, timing and pass criteria. It measures forward/lateral/yaw tracking,
independent arm motion, command expiry, explicit stopping, reset repeatability,
and contact with an immovable barrier. The barrier is fixed kinematic scene
geometry with contact reporting, not a dynamic object receiving reset velocities.

Collision means nonzero measured force on the barrier, with no base passage
through it; ground contacts are excluded because the barrier is immovable and
no other movable objects are present. This checks collision response, not obstacle
avoidance. The policy can continue pushing until stopped. It is not a navigation
controller.

Reports include policy/robot hashes, resolved configuration, joint order,
sampled physical observations, velocity errors, arm angles, stop speeds, contact
force and episode events. Failed assertions write a failed report and exit
nonzero, including when Isaac Sim uses fast shutdown.

The deployed policy (since 2026-10-22) observes the arm and was trained on flat ground with the
arm moving between the folded posture and library poses (out in 25% of episodes), left-right
symmetry and a base height of at least 0.56 m (`scripts/spot_sim_policy`, task
`Isaac-Velocity-Flat-SpotArmCam-ArmFineHeight-v0`, run `2026-10-21_18-05-44_h56w500`, model 13750).
At 250 Hz, 15 seeds, against the tuned 2026-10-19 policy (`spot_arm_w_cam_flat_tuned`, model 5000,
the reference of the stand work, never deployed here; `output_check/20261021-1859_h56_15seed`): no
metric worse and no falls; 1 s moves end 2.8 vs 4.1 cm off, navigation reaches 90 vs 77% of goals,
walking tilt with the arm swinging 2.2 vs 7.9 deg; it stands at 0.535 m after a stop. Its
command dead band (`command_dead_band` in `robots/spot/locomotion/locomotion.yaml`, measured 2026-10-22) raises
commands above 0.04 and up to 0.08 m/s (rad/s) to 0.08 in every direction, the slowest
direction's lowest walking speed (right), and sends commands at or below 0.04 as 0. The previously deployed policy (2026-10-16 to 2026-10-22, `output_check/20261021-2043_previous_walking_policy`)
was trained with the arm in library poses (`scripts/reinforcement_learning/spot_arm_locomotion`). CPU verification on 2026-10-16
measured velocity errors of 0.028 m/s for 0.4 m/s forward, 0.005 m/s for 0.3 m/s lateral, and
0.006 rad/s for 0.4 rad/s turning. Horizontal speeds after stopping were below 0.02 m/s. Below
its dead band (0.05-0.07 m/s, 0.10 rad/s) it crept in no set direction. The project runs at 250 Hz only (from
2026-10-17; the 500 Hz settings were removed on 2026-10-22): the transit footprint
`phygs/robots/spot/navigation/transit.json` is measured with the deployed policy at 250 Hz. With the
deployed policy 21 of 23 robot tests pass (`output_check/20261022-1738_solver8_all`); the two
Nav2 detours (known map, dynamic obstacle) pass after the RPP tuning and a 60 s budget
(`20261022-1929_nav2_tuned`, `20261022-1953_nav2_dynamic_budget60_r1`, `..._r2`).
These tests do not establish loaded-arm performance, external-force robustness, stairs,
uneven terrain, or arbitrary-speed capability.
