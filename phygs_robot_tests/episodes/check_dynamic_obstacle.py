# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle avoidance on the camera map, as an agent meets it: the plan walks to a goal while the
test moves a box into the route, then turns to face open floor, where the test puts the box in view
and takes it away. Nav2 maps the navigation cameras' point clouds as in every episode without
``known_map`` (spec ``nav_dynamic``, course ``phygs_robot_tests/scenes/dynamic_course``).

The test side checks that the box reached Nav2's costmap, the route was replanned, the footprint
never touched the box, the walk ended blocked or at the goal with the robot stopped, and that the box
put in view was marked and, once taken away, cleared. Thresholds: ``configs/dynamic_obstacle.yaml``.

    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/episodes/check_dynamic_obstacle.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_dynamic_obstacle
"""

from __future__ import annotations

import math

import numpy as np
import yaml

from phygs_robot_tests.episodes import episode

SETTINGS = yaml.safe_load((episode.CONFIGS / "dynamic_obstacle.yaml").read_text())
LETHAL = 253  # Nav2 costs at or above this: the robot's centre may not be there.
UNKNOWN = 255  # Not seen yet: passable for planning (Smac allow_unknown), neither marked nor free.


def marked(cost: int) -> bool:
    """An obstacle cell (253 or 254), not an unknown one."""
    return LETHAL <= cost < UNKNOWN


def plan(session, steps: episode.Steps) -> None:
    """Walk to the goal, turn to face +y, then stand while the test moves the box."""
    steps.enter("route")
    command = session.navigation.navigate_to(*SETTINGS["goal"], timeout=SETTINGS["budget_s"])
    steps.results["planned"] = session.navigation.feedback(command).status
    feedback = session.wait(command)
    steps.results["route"] = dict(status=feedback.status, reason=feedback.reason, replans=feedback.replans)
    steps.enter("route done")
    clearing = SETTINGS["clearing"]
    steps.enter("turn")
    session.wait(
        session.locomotion.command_velocity(0.0, 0.0, clearing["turn_yaw_radps"], duration=clearing["turn_seconds"])
    )
    session.sleep(clearing["stop_seconds"])
    steps.enter("box in view")
    session.sleep(clearing["wait_seconds"])
    steps.enter("box away")
    session.sleep(clearing["wait_seconds"])
    steps.enter("end")


def main() -> None:
    check, args, spec = episode.start(__doc__, "nav_dynamic")
    check.data.update(settings=SETTINGS)

    def body() -> None:
        with episode.environment(args, spec) as environment:
            nav2 = environment.control.navigation.backend  # Trusted diagnostic only: the costmaps.
            walls = [
                (np.asarray(c, dtype=float), np.asarray(s, dtype=float))
                for c, s in environment.scene_def.map_boxes(exclude=("obstacle",))
            ]
            _, size = episode.body_box(environment, "obstacle")
            every = max(1, round(SETTINGS["sample_period_s"] / environment.control.runtime.physics_dt))
            seen = dict(marked=[], route_cost=0, clearance=math.inf, marked_after=None, cleared_after=None)

            def snapshot(environment, step: str) -> dict:
                pose = environment.truth().base_pose()
                yaw = math.atan2(pose[1, 0], pose[0, 0])
                speed = float(np.linalg.norm(environment.scene["robot"].data.root_lin_vel_b.torch[0, :2].cpu().numpy()))
                taken = dict(time_s=environment.control.runtime.time_seconds, yaw=yaw, speed=speed)
                if step == "route":  # The walk started: move the box into it.
                    taken["cost_before"] = nav2.costmap_cost(*SETTINGS["blocked_position"][:2])
                    episode.move_body(environment, "obstacle", SETTINGS["blocked_position"])
                elif step == "box in view":  # Facing open floor: the box ahead, its near face's cell.
                    clearing = SETTINGS["clearing"]
                    ahead, direction = clearing["distance_m"], np.array([math.cos(yaw), math.sin(yaw)])
                    face = ahead - size[1] / 2 + 0.05  # Just behind the face the cameras see (box 1.2 m deep).
                    seen["cell"] = tuple(map(float, pose[:2, 3] + direction * face))
                    taken["cost_free"] = nav2.costmap_cost(*seen["cell"])
                    center = pose[:2, 3] + direction * ahead
                    episode.move_body(environment, "obstacle", (*center, SETTINGS["blocked_position"][2]))
                elif step == "box away":
                    episode.move_body(environment, "obstacle", SETTINGS["clearing"]["parked_position"])
                return taken

            def each_tick(environment, step: str) -> None:
                runtime = environment.control.runtime
                if runtime.tick % 5 == 0:  # The footprint against the box and the walls, every 0.02 s.
                    box = episode.body_box(environment, "obstacle")
                    seen["clearance"] = min(seen["clearance"], episode.clearance(environment, [box, *walls]))
                if runtime.tick % every:
                    return
                if step == "route":
                    cost = nav2.costmap_cost(*SETTINGS["blocked_position"][:2])
                    seen["route_cost"] = max(seen["route_cost"], cost if marked(cost) else 0)
                    box_center, box_size = episode.body_box(environment, "obstacle")
                    seen["marked"].append(
                        (round(runtime.time_seconds, 2), *episode.box_cells_marked(nav2, box_center, box_size))
                    )
                elif step == "box in view" and seen["marked_after"] is None:
                    if marked(nav2.costmap_cost(*seen["cell"])):
                        seen["marked_after"] = runtime.time_seconds - steps.snapshots["box in view"]["time_s"]
                elif step == "box away" and seen["cleared_after"] is None:
                    if nav2.costmap_cost(*seen["cell"]) < LETHAL:  # Seen free again.
                        seen["cleared_after"] = runtime.time_seconds - steps.snapshots["box away"]["time_s"]

            steps = episode.Steps(snapshot, each_tick)
            report = environment.run(
                lambda session: plan(session, steps), name="check_dynamic_obstacle.plan", during=steps
            )
            snaps = steps.snapshots
            check.data.update(results=steps.results, box_cells_marked=seen["marked"])
            episode.agent_checks(check, report, levels=("skills", "motion"))

            planned = steps.results.get("planned")
            check.measure("route_planned", planned, planned == "active", "Nav2 did not plan the free route")
            before = snaps["route"]["cost_before"]
            check.measure("cost_before_box", before, not marked(before), "the route was blocked before the box moved")
            check.measure(
                "box_in_costmap", seen["route_cost"], marked(seen["route_cost"]), "the box never reached the costmap"
            )
            route = steps.results.get("route", {})
            check.measure(
                "replans", route.get("replans"), (route.get("replans") or 0) >= 1, "the route was not replanned"
            )
            ok = route.get("status") in {"blocked", "succeeded"}
            check.measure("route_status", route, ok, "the walk did not end blocked or at the goal")
            speed = snaps["route done"]["speed"]
            check.measure("stopped_mps", round(speed, 3), speed <= SETTINGS["stopped_speed_mps"], "still moving")
            check.measure(
                "min_clearance_m", round(seen["clearance"], 3), seen["clearance"] > 0.0, "the footprint touched a box"
            )

            clearing = SETTINGS["clearing"]
            yaw = snaps["box in view"]["yaw"]
            check.measure(
                "turned_rad", round(yaw, 3), abs(yaw - math.pi / 2) < clearing["heading_tolerance_rad"], "not facing +y"
            )
            free = snaps["box in view"]["cost_free"]
            check.measure("cell_free_before", free, not marked(free), "the cell was marked before the box came")
            appeared, cleared = seen["marked_after"], seen["cleared_after"]
            check.measure(
                "marked_after_s", appeared, appeared is not None, f"not marked within {clearing['wait_seconds']} s"
            )
            check.measure(
                "cleared_after_s", cleared, cleared is not None, f"not cleared within {clearing['wait_seconds']} s"
            )

    check.run(body)


if __name__ == "__main__":
    main()
