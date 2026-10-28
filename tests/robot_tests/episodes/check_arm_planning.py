# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arm planning at the grid table, as an agent drives it: look at the target, move the hand over the
clutter box and over a free spot, cancel a planned move, stow, then walk a little with the cameras and
the planner running. The spec decides what the planner avoids: the scene's known boxes
(``table_reach``) or depth maps from the cameras (``table_reach_depth``); ``--planner`` picks cuRobo
or MoveIt.

The test side checks the views, the reached poses, the cancel, the stow and that no arm link touched
the furniture, plus the planner itself: its robot model matches the physical links, it refuses a
start inside an obstacle, and every camera frame arrived. Thresholds: ``configs/arm_planning.yaml``.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_arm_planning.py \
        --spec scripts/PhyGS/tests/robot_tests/configs/episodes/table_reach_depth.yaml \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_arm_planning_depth
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import yaml
from phygs_robot_tests.episodes import episode

from phygs.episodes.session import AgentFailed, PointTarget

SETTINGS = yaml.safe_load((episode.CONFIGS / "arm_planning.yaml").read_text())


def arm_joints(session) -> dict[str, float]:
    """The measured arm joint angles [rad] (the gripper left out), as the agent observes them."""
    joints = session.observe().joints
    arm = set(session.robot.arm_joints)
    return {
        name: float(value) for name, value in zip(joints.names, joints.positions.tolist(), strict=True) if name in arm
    }


def plan(session, steps: episode.Steps) -> None:
    """The look, the moves, the cancel, the stow and the walk; feedback goes to ``steps.results``."""
    target = session.hints.target()
    point = (target.center[0], target.center[1], target.center[2] + target.size[2] / 2)
    steps.results["look_point"] = point
    for distance, elevation in SETTINGS["look_views"]:  # As an agent would: the next view if refused.
        steps.enter(f"look {distance} m {elevation} deg")
        feedback = session.wait(
            session.manipulation.look_at(PointTarget(point), distance_m=distance, elevation_deg=elevation)
        )
        steps.results.setdefault("look", []).append(dict(view=[distance, elevation], **asdict(feedback)))
        if feedback.status == "succeeded":
            break
    else:
        raise AgentFailed("look_at: every requested view was refused")
    steps.enter("look hold")  # The hand camera captures after the arm settled.
    session.sleep(SETTINGS["look_hold_seconds"])
    steps.enter("look held")

    for name, pose in SETTINGS["poses"].items():
        steps.enter(f"move {name}")
        feedback = session.wait(session.manipulation.move_to_pose(tuple(pose), tool="hand"))
        steps.results[f"move {name}"] = asdict(feedback)
        if feedback.status != "succeeded":
            raise AgentFailed(f"move_to_pose {name}: {feedback.reason}")
        steps.enter(f"at {name}")

    cancel = SETTINGS["cancel"]
    goal = list(session.manipulation.tool_pose("wrist"))
    goal[2] += cancel["lift_m"]
    steps.enter("cancel")
    command = session.manipulation.move_to_pose(tuple(goal), tool="wrist")
    session.sleep(cancel["after_seconds"])
    before = arm_joints(session)
    session.manipulation.cancel(command)
    feedback = session.wait(command)
    session.sleep(cancel["settle_seconds"])
    after, velocities = arm_joints(session), session.observe().joints
    speed = max(
        abs(float(v)) for n, v in zip(velocities.names, velocities.velocities.tolist(), strict=True) if n in before
    )
    steps.results["cancel"] = dict(
        status=feedback.status,
        drift_rad=max(abs(after[name] - before[name]) for name in before),
        speed_radps=speed,
    )

    steps.enter("stow")
    feedback = session.wait(session.manipulation.stow())
    steps.results["stow"] = asdict(feedback)
    if feedback.status != "succeeded":
        raise AgentFailed(f"stow: {feedback.reason}")

    walk = SETTINGS["walk"]
    steps.enter("walk")
    session.wait(session.locomotion.command_velocity(walk["speed_mps"], 0.0, 0.0, duration=walk["seconds"]))
    steps.enter("walked")
    session.sleep(0.5)


def snapshot(environment, step: str) -> dict:
    """At each step's first tick: the time, the base, the wrist, and (once the planner runs) its
    model against the physical links."""
    base = environment.truth().base_pose()
    taken = dict(
        time_s=round(environment.control.runtime.time_seconds, 3),
        base_xy=base[:2, 3].tolist(),
        wrist=episode.wrist_pose(environment),
        planner_result=(environment.planner.last_plan or {}).get("result"),
    )
    if environment.planner.model:
        taken["alignment"] = episode.link_alignment(environment)
    if step == "look held":
        taken["camera"] = hand_view(environment)
    return taken


def hand_view(environment) -> dict:
    """The look camera's latest frame (the one look_at aims): capture time [s] and its world pose (4x4)."""
    packet = environment.control.observe()
    camera = packet.cameras[environment.robot.look_camera]
    pose = packet.transform("world", camera.frame, camera.stamp).cpu().numpy()
    return dict(stamp_s=camera.stamp.time_seconds, pose=pose, rgb=camera.rgb.cpu().numpy())


def wrist_goal(environment, hand_goal) -> tuple[float, ...]:
    """The wrist pose [x, y, z, qx, qy, qz, qw] that puts the robot's hand tool at ``hand_goal``."""
    from phygs_robot_tests.routines.geometry import matrix, xyzw

    return xyzw(matrix(hand_goal) @ np.linalg.inv(environment.robot.tool("hand")))


def view_errors(camera_pose: np.ndarray, point) -> tuple[float, float, float]:
    """Distance [m], elevation [deg] and axis angle [deg] of the camera to a point."""
    offset = np.asarray(point) - camera_pose[:3, 3]
    distance = float(np.linalg.norm(offset))
    elevation = float(np.degrees(np.arcsin(-offset[2] / distance)))
    angle = float(np.degrees(np.arccos(np.clip(camera_pose[:3, 2] @ offset / distance, -1.0, 1.0))))
    return distance, elevation, angle


def main() -> None:
    check, args, spec = episode.start(
        __doc__,
        "table_reach",
        lambda flags: flags.add_argument("--planner", choices=("curobo", "moveit"), default="curobo"),
    )
    check.data.update(planner=args.planner, known_map="known_map" in spec.grants, settings=SETTINGS)

    def body() -> None:
        with episode.environment(args, spec, planner=args.planner) as environment:
            audit = episode.geometry_audit(environment, SETTINGS["geometry_cell_size"])
            check.data["geometry_audit"] = audit
            check.measure("geometry_audit_qualified", audit["qualified"], audit["qualified"], "robot geometry coverage")
            cameras = episode.CameraTiming(environment, environment.robot.cameras_for("planning"))
            render = environment.render_ticks

            def each_tick(environment, step: str) -> None:
                if environment.control.runtime.tick % render == 0:
                    cameras(environment)

            steps = episode.Steps(snapshot, each_tick)
            report = environment.run(lambda session: plan(session, steps), name="check_arm_planning.plan", during=steps)
            check.data.update(results=steps.results, commands=len(report["commands"]))
            episode.agent_checks(check, report, levels=("skills", "motion"))
            snaps = steps.snapshots

            # The view: the camera's frame after the hold, at the requested distance and elevation.
            view = snaps["look held"]["camera"]
            reached = steps.results["look"][-1]["view"]
            distance, elevation, angle = view_errors(view["pose"], steps.results["look_point"])
            episode.save_view(view["rgb"], args.output / "look.png", f"look_at: {distance:.2f} m, {elevation:.0f} deg")
            tolerance = SETTINGS["look_tolerance"]
            fresh = view["stamp_s"] > snaps["look hold"]["time_s"]
            check.measure("look_frame_after_move", round(view["stamp_s"], 3), fresh, "no hand frame after the move")
            check.measure("look_distance_m", round(distance, 4), abs(distance - reached[0]) <= tolerance["distance_m"])
            check.measure(
                "look_elevation_deg", round(elevation, 2), abs(elevation - reached[1]) <= tolerance["elevation_deg"]
            )
            check.measure("look_axis_deg", round(angle, 2), angle <= tolerance["axis_deg"], "camera axis off the point")

            # The reached poses, the cancel, the stow.
            tolerance = SETTINGS["pose_tolerance"]
            for name, goal in SETTINGS["poses"].items():  # The wrist where the hand goal puts it.
                position, rotation = episode.pose_error(snaps[f"at {name}"]["wrist"], wrist_goal(environment, goal))
                ok = position <= tolerance["position_m"] and rotation <= tolerance["rotation_rad"]
                check.measure(f"pose_{name}_error", dict(m=round(position, 4), rad=round(rotation, 4)), ok)
            cancel, limits = steps.results["cancel"], SETTINGS["cancel"]
            check.measure("cancel_status", cancel["status"], cancel["status"] == "cancelled", "not cancelled")
            check.measure("cancel_drift_rad", round(cancel["drift_rad"], 4), cancel["drift_rad"] <= limits["drift_rad"])
            check.measure(
                "cancel_speed_radps", round(cancel["speed_radps"], 4), cancel["speed_radps"] <= limits["speed_radps"]
            )
            error = environment.truth().stow_error_rad()
            check.measure("stow_error_rad", round(error, 4), error <= SETTINGS["stow_tolerance_rad"], "not stowed")
            walked = float(np.linalg.norm(np.subtract(snaps["walked"]["base_xy"], snaps["walk"]["base_xy"])))
            check.measure("walk_distance_m", round(walked, 3), walked >= SETTINGS["walk"]["minimum_distance_m"])

            # The planner: its model on the physical links at every step, a blocked start refused.
            fk = SETTINGS["fk_tolerance"]
            aligned = [snap["alignment"] for snap in snaps.values() if "alignment" in snap]
            worst = max(aligned, key=lambda item: item["position_m"])
            rotation = max(item["rotation_rad"] for item in aligned)
            ok = worst["position_m"] <= fk["position_m"] and rotation <= fk["rotation_rad"]
            check.measure("planner_link_alignment", dict(worst, rotation_rad=rotation), ok, "model off the links")
            raised = dict(zip(environment.robot.arm.joints, environment.robot.planning.default_joints, strict=True))
            refused = episode.blocked_start(environment, raised)
            check.data["blocked_start"] = refused
            check.measure("blocked_start_refused", not refused["success"], not refused["success"], "accepted")

            # The cameras and the furniture.
            timing = cameras.result()
            ok = (
                not timing["invalid"]
                and timing["max_gap_s"] is not None
                and timing["max_gap_s"] <= cameras.period + 1e-6
            )
            check.measure("camera_frames", timing, ok, "a camera frame was invalid or a capture was skipped")
            touched = episode.unintended_contacts(report)
            check.measure("arm_contacts", len(touched), not touched, f"first {touched[:1]}")

    check.run(body)


if __name__ == "__main__":
    main()
