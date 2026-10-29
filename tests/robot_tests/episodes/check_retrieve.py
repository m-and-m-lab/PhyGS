# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retrieve: at the end the target is in the hand and the arm stowed.

The plan: stow; if the target is in a drawer, walk to the drawer, open it and walk to where the
robot reaches into it, otherwise walk to the furniture it rests on; pick it and stow holding it.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_retrieve.py \
        --spec scripts/PhyGS/tests/robot_tests/configs/episodes/dev_grid_drawer.yaml \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_retrieve_grid_drawer
"""

from __future__ import annotations

from phygs_robot_tests.episodes import episode
from phygs_robot_tests.routines import arm, base, drawer

from phygs.session import AgentFailed


def plan(session) -> None:
    """Hold the target in the gripper with the arm stowed."""
    arm.stow(session)
    target = session.hints.target()
    if target.inside is not None:
        base.go_to(session, target.inside, "open")
        arm.look_at(session, session.hints.part(target.inside).handle_center, "handle")
        drawer.open_drawer(session, target.inside)
        arm.stow(session)
        target = session.hints.target()  # The open drawer moved it.
        base.go_to(session, target.inside, "reach_in", target.center)
    else:
        base.go_to(session, target.on, "reach")
    held = arm.pick(session, target)
    arm.stow(session, held)
    if not session.manipulation.holding:
        raise AgentFailed("The target dropped in the stow.")


def main() -> None:
    check, args, spec = episode.start(__doc__, "dev_grid_drawer")

    def body() -> None:
        if spec.task["goal"] != "retrieve":
            raise ValueError(f"{spec.id} is not a retrieve episode.")
        with episode.world(args, spec) as world:
            report = world.run(plan, name="check_retrieve.plan")
            episode.agent_checks(check, report)
            evaluation = report["evaluation"]  # The evaluator's result from ground truth, after the stop.
            checks, measured = evaluation["checks"], evaluation["measured"]
            check.measure("target_held", dict(height_m=measured["target_height_m"]), checks["held"], "not in the hand")
            check.measure("stow_error_rad", measured["stow_error_rad"], checks["stowed"], "the arm is not stowed")
            check.measure("evaluation", evaluation["outcome"], evaluation["outcome"] == "success", evaluation["reason"])
            check.data["stopped"] = report["stopped"]

    check.run(body)


if __name__ == "__main__":
    main()
