# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation on the known map, as an agent drives it: ``navigate_to`` goals that must be reached,
refused or cancelled, and one goal reached with body velocities alone. The cases per spec are in
``phygs/configs/navigation.yaml``: the course (``nav_course``: around a wall, to a heading, into the wall,
into a closed pen) and the house (``nav_house``: into the kitchen island, two goals, a cancel).

The test side judges arrival from ground truth (the robot within the goal's tolerance and stopped,
``PointNavigationEvaluator``), checks that the robot stopped after every walk, that its footprint
never overlapped a box, and that a velocity command is refused while navigation drives. Nav2 runs
as the episode harness starts it (``phygs/configs/environment.yaml``, ``phygs/configs/navigation/``).

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_navigation.py \
        --spec scripts/PhyGS/tests/robot_tests/configs/episodes/nav_course.yaml \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_navigation
"""

from __future__ import annotations

import math

import numpy as np
import yaml
from phygs_robot_tests.episodes import episode

SETTINGS = yaml.safe_load((episode.CONFIGS / "navigation.yaml").read_text())
DEFAULT_TOLERANCE = (0.04, 0.04)  # navigate_to's position [m] and heading [rad] tolerances.


def heading_error(goal: float, yaw: float) -> float:
    """Signed smallest angle [rad] from ``yaw`` to ``goal``."""
    return math.remainder(goal - yaw, 2 * math.pi)


def drive(session, goal, tolerance) -> str:
    """Walk to ``goal`` (x, y [m], heading [rad]) with body velocity commands alone: straight at it,
    turning to the heading on the way, then stop. Returns ``succeeded`` or ``timed_out``."""
    limits = SETTINGS["velocity"]
    end = session.time() + SETTINGS["case_budget_s"]
    while session.time() < end:
        pose = session.observe().localization.pose.matrix
        x, y, yaw = float(pose[0, 3]), float(pose[1, 3]), math.atan2(float(pose[1, 0]), float(pose[0, 0]))
        dx, dy = goal[0] - x, goal[1] - y
        distance = math.hypot(dx, dy)
        if distance <= tolerance[0]:
            session.wait(session.locomotion.stop())
            return "succeeded"
        speed = min(limits["max_speed_mps"], distance)
        vx, vy = speed * dx / distance, speed * dy / distance  # World frame.
        c, s = math.cos(yaw), math.sin(yaw)
        turn = max(-limits["max_yaw_radps"], min(limits["max_yaw_radps"], heading_error(goal[2], yaw)))
        session.locomotion.command_velocity(c * vx + s * vy, -s * vx + c * vy, turn, duration=2 * limits["period_s"])
        session.sleep(limits["period_s"])
    session.wait(session.locomotion.stop())
    return "timed_out"


def plan(session, steps: episode.Steps, cases: list[dict], start, returns: bool) -> None:
    """Run the cases in order; each one's final navigation feedback goes to ``steps.results``."""
    for index, case in enumerate(cases):
        name, goal = case["name"], case["goal"]
        if returns and index:
            steps.enter(f"return before {name}")
            feedback = session.wait(session.navigation.navigate_to(*start, timeout=SETTINGS["case_budget_s"]))
            steps.results[f"return before {name}"] = feedback.status
        steps.enter(name)
        tolerance = case.get("tolerance", DEFAULT_TOLERANCE)
        if case.get("by") == "velocity":
            steps.results[name] = dict(status=drive(session, goal, tolerance))
        else:
            command = session.navigation.navigate_to(
                *goal,
                position_tolerance=tolerance[0],
                heading_tolerance=tolerance[1],
                timeout=SETTINGS["case_budget_s"],
            )
            if "cancel_after_s" in case:
                session.sleep(case["cancel_after_s"])
                try:  # Navigation owns the legs: a direct velocity command must be refused.
                    session.locomotion.command_velocity(0.0, 0.0, 0.0)
                    steps.results[f"{name} velocity refused"] = None
                except RuntimeError as error:
                    steps.results[f"{name} velocity refused"] = str(error)
                session.navigation.cancel(command)
            feedback = session.wait(command)
            steps.results[name] = dict(status=feedback.status, reason=feedback.reason, replans=feedback.replans)
        steps.enter(f"{name} done")


def main() -> None:
    check, args, spec = episode.start(__doc__, "nav_course")
    if spec.id not in SETTINGS["cases"]:
        raise SystemExit(f"phygs/configs/navigation.yaml has no cases for {spec.id}")
    cases, returns = SETTINGS["cases"][spec.id], spec.id in SETTINGS["return_to_start"]
    check.data.update(cases=cases)

    def body() -> None:
        with episode.world(args, spec) as world:
            robot = world.scene["robot"]
            matches = episode.footprint_matches(world)
            check.measure("footprint_profile_matches_robot", matches, matches, "re-measure the footprint (rung 4)")
            boxes = [(np.asarray(c, dtype=float), np.asarray(s, dtype=float)) for c, s in world.scene_def.map_boxes()]
            every = max(1, round(SETTINGS["clearance_period_s"] / world.control.runtime.physics_dt))
            by_name = {case["name"]: case for case in cases}
            evaluators, turned, closest = {}, {}, [math.inf]

            def snapshot(world, step: str) -> dict:
                sample = episode.physical_sample(world)
                if step in by_name:  # A case starts: its physical evaluator.
                    case = by_name[step]
                    tolerance = case.get("tolerance", DEFAULT_TOLERANCE)
                    budget = SETTINGS["case_budget_s"] + 1.0
                    evaluators[step] = episode.navigation_evaluator(
                        spec, case["goal"], tolerance, budget, sample.time, world.robot.fallen_height
                    )
                    turned[step] = (sample.yaw, 0.0)
                return dict(x=sample.x, y=sample.y, yaw=sample.yaw, speed=sample.speed, yaw_speed=sample.yaw_speed)

            def each_tick(world, step: str) -> None:
                if world.control.runtime.tick % every:
                    return
                here = robot.data.root_pos_w.torch[0, :2].cpu().numpy()
                near = [(c, s) for c, s in boxes if np.hypot(*(c[:2] - here)) < 3.0 + np.max(s[:2])]  # Within reach.
                closest[0] = min(closest[0], episode.clearance(world, near))
                if step in evaluators:
                    sample = episode.physical_sample(world)
                    evaluators[step].update(sample)
                    yaw0, most = turned[step]
                    turned[step] = (yaw0, max(most, abs(heading_error(sample.yaw, yaw0))))

            steps = episode.Steps(snapshot, each_tick)
            start = tuple(spec.robot.start)
            report = world.run(
                lambda session: plan(session, steps, cases, start, returns), name="check_navigation.plan", during=steps
            )
            check.data.update(results=steps.results, evaluators={k: v.result() for k, v in evaluators.items()})
            episode.agent_checks(check, report, levels=("skills", "motion"))
            stopped = SETTINGS["stopped"]
            for case in cases:
                name, result = case["name"], steps.results.get(case["name"], {})
                status = result.get("status")
                check.measure(f"{name}_status", result, status in case["expect"], f"expected {case['expect']}")
                done = steps.snapshots[f"{name} done"]
                if status == "succeeded":
                    evaluator = evaluators[name]
                    check.measure(
                        f"{name}_arrived", evaluator.status, evaluator.status == "succeeded", "not physically there"
                    )
                    tolerance = case.get("tolerance", DEFAULT_TOLERANCE)
                    error = math.hypot(done["x"] - case["goal"][0], done["y"] - case["goal"][1])
                    turn = abs(heading_error(case["goal"][2], done["yaw"]))
                    ok = error <= tolerance[0] and turn <= tolerance[1]
                    check.measure(f"{name}_final_error", dict(m=round(error, 4), rad=round(turn, 4)), ok)
                if status in {"succeeded", "cancelled", "blocked"}:
                    still = done["speed"] < stopped["speed_mps"] and abs(done["yaw_speed"]) < stopped["yaw_radps"]
                    value = dict(mps=round(done["speed"], 3), radps=round(done["yaw_speed"], 3))
                    check.measure(f"{name}_stopped", value, still, "still moving after the walk ended")
                if case.get("turns"):
                    most = turned[name][1]
                    ok = most > SETTINGS["turn_min_rad"]
                    check.measure(f"{name}_turned_rad", round(most, 3), ok, "did not face its travel")
                if "cancel_after_s" in case:
                    refused = steps.results.get(f"{name} velocity refused")
                    ok = refused is not None and "Cancel navigation" in refused
                    check.measure(f"{name}_velocity_refused", refused, ok, "direct motion bypassed navigation")
                if evaluators.get(name) is not None:
                    falls = evaluators[name].falls
                    check.measure(f"{name}_falls", falls, falls == 0, "the robot fell")
            returned = {key: value for key, value in steps.results.items() if key.startswith("return before")}
            ok = all(value == "succeeded" for value in returned.values())
            check.measure("returns_to_start", returned, ok, "a walk back to the start failed")
            check.measure("min_clearance_m", round(closest[0], 3), closest[0] > 0.0, "the footprint overlapped a box")

    check.run(body)


if __name__ == "__main__":
    main()
