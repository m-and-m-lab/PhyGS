# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick and place at the grid table, as an agent drives it: pick the target next to the clutter box
straight down, hold it, carry it over the clutter and back, set it down on a free spot, stow. With
the scene's known boxes (``table_reach``) a carry goal that only the held object would hit must be
refused first; on depth (``table_reach_depth``) the clutter's top is unseen while holding (the
front cameras sit below it and the held box hides it from the hand camera), so that refusal is not
asked. ``--planner`` picks cuRobo or MoveIt.

The plan is the routines' pick and place (``routines/arm.py``): the starting point for a scripted
manipulation agent. The test side checks the grasp, the lift, that neither the arm nor the held
object touched the furniture, where the object rests and the stow. Thresholds: ``configs/pick_place.yaml``.

    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/episodes/check_pick_place.py \
        --spec scripts/PhyGS/phygs_robot_tests/configs/episodes/table_reach.yaml \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_pick_place
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import yaml

from phygs.episodes.session import AgentFailed
from phygs_robot_tests.episodes import episode
from phygs_robot_tests.routines import PARAMS, arm
from phygs_robot_tests.routines.geometry import matrix, xyzw

SETTINGS = yaml.safe_load((episode.CONFIGS / "pick_place.yaml").read_text())
CARRY = ("refused carry", "after refusal", "carry over clutter", "carry back")  # The held object must not touch.


def box_of(boxes: dict, name: str):
    """A body's box (center, size) [m] among the hints' collider boxes (keyed by collider path)."""
    return next((np.asarray(c), np.asarray(s)) for path, (c, s) in boxes.items() if f"/{name}/" in path)


def plan(session, steps: episode.Steps, known_map: bool) -> None:
    """Pick, hold, (refuse,) carry, place, stow; feedback goes to ``steps.results``."""
    target = session.hints.target()
    steps.enter("pick")
    held = arm.pick(session, target, [None], look=not known_map)  # On depth, look first: the plan maps what it saw.
    steps.enter("picked")
    session.sleep(SETTINGS["hold_seconds"])
    steps.enter("held")

    if known_map:  # The clutter and the table from the known boxes.
        boxes = session.hints.boxes()
        (cx, cy, cz), clutter_size = box_of(boxes, "clutter")
        clutter_top = cz + clutter_size[2] / 2
        table_center, table_size = box_of(boxes, "table")
        table_top = float(table_center[2] + table_size[2] / 2)
    else:  # No known boxes: the layout as built; an agent that perceives would measure it.
        layout = SETTINGS["layout"]
        (cx, cy), clutter_top, table_top = layout["clutter_xy"], layout["clutter_top_m"], layout["table_top_m"]
    wrist = matrix(session.manipulation.tool_pose("wrist"))
    corners = np.array(np.meshgrid(*[[-s / 2, s / 2] for s in held.size])).reshape(3, -1).T
    box = wrist @ held.wrist_from_object
    bottom = float(np.min(corners @ box[:3, :3].T[:, 2] + box[2, 3]))  # The held object's lowest point.

    def over_clutter(bottom_z: float) -> tuple[float, ...]:
        """The wrist over the clutter with the object's bottom at ``bottom_z`` [m]."""
        return (float(cx), float(cy), float(wrist[2, 3] + bottom_z - bottom), *xyzw(wrist)[3:])

    if known_map:
        steps.enter("refused carry")
        goal = over_clutter(clutter_top - SETTINGS["refused_depth_m"])
        feedback = session.wait(session.manipulation.move_to_pose(goal, held=held, tool="wrist"))
        steps.results["refused carry"] = asdict(feedback)
        steps.enter("after refusal")
    for name, goal in (
        ("carry over clutter", over_clutter(clutter_top + SETTINGS["carry_clearance_m"])),
        ("carry back", xyzw(wrist)),
    ):
        steps.enter(name)
        feedback = session.wait(session.manipulation.move_to_pose(goal, held=held, tool="wrist"))
        steps.results[name] = asdict(feedback)
        if feedback.status != "succeeded":
            raise AgentFailed(f"{name}: {feedback.reason}")

    steps.enter("place")
    spot = (*SETTINGS["place_xy"], table_top)
    steps.results["place_spot"] = spot
    arm.place(session, spot, held)
    steps.enter("placed")
    session.sleep(SETTINGS["hold_seconds"])  # The released object settles.
    steps.enter("settled")
    arm.stow(session)


def gap(a, b) -> float:
    """Separation [m] of two axis-aligned boxes (center, size); negative: they overlap."""
    (ca, sa), (cb, sb) = a, b
    return float(np.max(np.abs(np.subtract(ca, cb)) - (np.asarray(sa) + np.asarray(sb)) / 2))


def main() -> None:
    check, args, spec = episode.start(
        __doc__,
        "table_reach",
        lambda flags: flags.add_argument("--planner", choices=("curobo", "moveit"), default="curobo"),
    )
    known_map = "known_map" in spec.grants
    check.data.update(planner=args.planner, known_map=known_map, settings=SETTINGS)
    target = spec.task["target"]

    def body() -> None:
        with episode.environment(args, spec, planner=args.planner) as environment:
            truth = environment.truth()
            start_center, size = environment.scene_def.initial_box(target)
            furniture = ("table", "clutter")
            closest = {name: math.inf for name in furniture}

            def snapshot(environment, step: str) -> dict:
                center, _ = episode.body_box(environment, target)
                taken = dict(wrist=episode.wrist_pose(environment), target_center=center)
                taken["planner_result"] = (environment.planner.last_plan or {}).get("result")
                if step == "picked":
                    taken["gripper_bottom"] = episode.jaw_bottom(environment)
                return taken

            def each_tick(environment, step: str) -> None:
                if step in CARRY:  # The held object against the furniture while carried.
                    held = episode.body_box(environment, target)
                    for name in furniture:
                        closest[name] = min(closest[name], gap(held, episode.body_box(environment, name)))

            steps = episode.Steps(snapshot, each_tick)
            report = environment.run(
                lambda session: plan(session, steps, known_map), name="check_pick_place.plan", during=steps
            )
            check.data.update(results=steps.results)
            episode.agent_checks(check, report)
            snaps = steps.snapshots

            # The pre-grasp: where the pick's planned move put the wrist before its line in (the first
            # wrist pose the pick reads). The pre-grasp has the robot's grasp frame above the object's
            # top: the wrist sits its approach offset straight above that, its other offsets along its
            # own gripper axes. After the grasp the wrist sits where the closing jaw left it; the grasp
            # is judged by the object below the jaws and lifted.
            arrival = episode.first_tool_pose(args.output)
            grasp = arm.tool("grasp")
            along, sideways, across = grasp[:3, :3].T @ grasp[:3, 3]  # In the gripper's axes.
            axes = arm.gripper_axes(arrival)
            pregrasp = start_center + [0.0, 0.0, size[2] / 2 + PARAMS["approach_m"]]
            wrist_there = pregrasp - arm.DOWN * along - axes[:, 1] * sideways - axes[:, 2] * across
            error = float(np.linalg.norm(arrival[:3, 3] - wrist_there))
            check.measure("pregrasp_error_m", round(error, 4), error <= SETTINGS["pregrasp_tolerance_m"])
            hang = snaps["picked"]["gripper_bottom"] - (snaps["picked"]["target_center"][2] - size[2] / 2)
            check.measure("hang_below_jaws_m", round(hang, 4), hang > SETTINGS["minimum_hang_m"], "object too high")
            lift = float(snaps["held"]["target_center"][2] - start_center[2])
            check.measure("lift_m", round(lift, 4), lift >= SETTINGS["minimum_lift_m"], "the target was not lifted")

            # The refusal (known boxes only): failed before the arm moved, no trajectory.
            if known_map:
                refused = steps.results["refused carry"]
                check.measure("refused_carry_status", refused["status"], refused["status"] == "failed", "accepted")
                result = snaps["after refusal"]["planner_result"]
                check.measure("refused_carry_plan", result, result != "planned", "a trajectory was planned")
                moved = snaps["after refusal"]["wrist"][:3, 3] - snaps["refused carry"]["wrist"][:3, 3]
                motion = float(np.linalg.norm(moved))
                check.measure("refused_carry_wrist_m", round(motion, 4), motion < SETTINGS["refused_wrist_motion_m"])

            # The carry: the held object never touched the furniture.
            for name in furniture:
                check.measure(f"carry_gap_{name}_m", round(closest[name], 4), closest[name] > 0.0, "touched")

            # The place: on the spot, on the table, upright; then the stow.
            pose = truth.object_pose(target)
            spot = np.asarray(steps.results["place_spot"])
            tolerance = SETTINGS["place_tolerance"]
            offset = float(np.linalg.norm(pose[:2, 3] - spot[:2]))
            height = float(pose[2, 3] - (spot[2] + size[2] / 2))
            tilt = float(np.degrees(np.arccos(np.clip(pose[2, 2], -1.0, 1.0))))
            check.measure("place_error_m", round(offset, 4), offset <= tolerance["position_m"])
            check.measure("place_height_error_m", round(height, 4), abs(height) <= tolerance["height_m"])
            check.measure("place_tilt_deg", round(tilt, 2), tilt <= tolerance["tilt_deg"])
            held_thresholds = environment.task.thresholds  # One definition of "held": the task's evaluator's.
            held = truth.held(target, lift_m=held_thresholds["held_lift_m"], reach_m=held_thresholds["held_reach_m"])
            check.measure("released", not held, not held, "still in the hand")
            error = truth.stow_error_rad()
            check.measure("stow_error_rad", round(error, 4), error <= SETTINGS["stow_tolerance_rad"], "not stowed")
            touched = episode.unintended_contacts(report, allowed=(target,))
            check.measure("arm_contacts", len(touched), not touched, f"first {touched[:1]}")

    check.run(body)


if __name__ == "__main__":
    main()
