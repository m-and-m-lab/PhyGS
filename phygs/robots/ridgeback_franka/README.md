# Ridgeback-Franka

Clearpath Ridgeback (an omnidirectional base) with a Franka Emika Panda arm and its two-finger hand:
Isaac Lab's asset (`isaaclab_assets.robots.ridgeback_franka`). It is the framework's second robot, in
bring-up. It differs from Spot where the framework had Spot-only assumptions: the base moves on
three velocity-driven joints under a fixed articulation root (no walking policy), the arm has seven
joints with other names, and the gripper is two prismatic fingers [m] that follow one primary joint.

| File | What |
| --- | --- |
| `robot.yaml` | The description. Values marked "to measure" or "to tune" come from the ladder. |
| `fetch.py` | Downloads the USD into `assets/` (gitignored), every file checked against a pinned hash. |
| `make_planning_urdf.py` | Writes `planning.urdf` from the USD's joint frames, axes and limits. |
| `planning.urdf` | The planners' model, rooted at `base_link`. |
| `meshes/` | Franka collision meshes from `franka_description` (Apache-2.0, `meshes/LICENSE`). |
| `actuators.py` | Isaac Lab's actuator groups for the asset. |
| `base/base.yaml` | The planar base's command shaping and stop (`PlanarJointBase`). |

```bash
python3 scripts/PhyGS/phygs/robots/ridgeback_franka/fetch.py
./isaaclab.sh -p scripts/PhyGS/phygs/robots/ridgeback_franka/make_planning_urdf.py
```

## Bring-up ladder

| Rung | Status |
| --- | --- |
| description | Passes (2026-10-28): `planning.urdf` and the USD match `robot.yaml` (`tests/test_robot_description.py`). The URDF's forward kinematics equal the USD's link poses at its authored joint angles. |
| planning model, offline | Builds (2026-10-28): collision spheres from the USD, the sphere audit qualified, IK solves. |
| bring-up | Passes (2026-10-28, `components/check_bring_up.py`, `output_check/20261028-1319_bring_up_ridgeback_franka`): spawns stowed; drives 0.637 m forward and sideways and turns 1.289 rad (2 s at 60 % of its limits), stops in 0.7 s; the arm holds the stow while driving and reaches the pre-stow; the fingers close to 0.0 m and open to 0.04 m; both cameras give depth; localization follows the base link. |
| 03 footprint | Measured (2026-10-28, `output_check/20261028-1155_ridgeback_footprint`): standing outline x -0.466 to 0.706 m, y +-0.395 m, padded 0.05 m; the stowed arm held through 1.0 m/s and 1.5 rad/s drives; stops within 0.5 m from 1.0 m/s. Adopted as `navigation/transit.json`. |
| 06 navigation_course | Passes (2026-10-28, `output_check/20261028-1205_ridgeback_nav_course`): around the wall, to a heading, by velocity commands, the goal in the wall refused (invalid_goal), the one in the closed pen unreachable, a cancel; arrivals within 1.9 cm, closest approach to a box 0.27 m. |
| 05 arm | Passes (2026-10-28, `output_check/20261028-1417_arm_rungs_two_robots/ridgeback_arm`): raise, open and close, sweep, a free-space wrist pose, cancel (1.2 mm drift), reset. |
| 08 arm_planning_known (cuRobo) | Passes (2026-10-28, `output_check/20261028-1440_ridgeback_arm_planning/ridgeback_arm_planning_known`, start x -0.12 m): look_at 0.50 m at 60 deg on the target, hand goals over the clutter and a free spot within 1.3 mm, cancel, stow, walk; the planner's model on the links (0 m), a blocked start refused, no arm contact. Needed: the fingers never checked against each other, cuRobo with no locked joints. |
| 08 arm_planning_known (MoveIt), 09 arm_planning_depth (cuRobo, MoveIt) | Pass (2026-10-28, `output_check/20261028-1911_ridgeback_arm_variants`): look_at 0.50 m at 60 deg, stow, a blocked start refused, no arm contact. |
| 10 pick_place (cuRobo, MoveIt) | Passes (2026-10-28, `output_check/20261028-1855_pick_place_two_robots`, `ridgeback_pick_place_{known,depth,known_moveit}`; depth with MoveIt in `output_check/20261028-1911_ridgeback_arm_variants`): the box picked straight down and lifted 0.19 m, the carry into the clutter refused, the carry over it with the box 0.9-2.0 cm clear, set down within 0.6 mm (MoveIt 2.2 mm), upright, then stowed; no arm contact. Needed: the plain Franka finger gains (the asset's 1e5 stiffness toppled the box), the `grasp` frame 3.7 cm inside the fingertips, the palm allowed to touch the held box (`held_contact_links`), the place spot at x 0.80 m (the old one's pre-place was 2.1 cm out of reach). |
| other rungs | Not run yet with `--robot ridgeback_franka` (drawers, retrieve: their stances per scene are Spot's). |

Known open points: the camera mounts are first guesses; the `handle` tool frame sits between the
fingertips until the drawer rungs measure it; the routines' stances per scene are Spot's.
