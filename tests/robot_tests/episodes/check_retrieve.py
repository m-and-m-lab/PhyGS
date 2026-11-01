# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retrieve: at the end the target is in the hand and the arm stowed.

The plan, with the framework's composite skills: stow; if the target is in a drawer, open it and
walk to where the robot reaches into it, otherwise walk to where it reaches the target; pick it and
stow holding it.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_retrieve.py \
        --spec scripts/PhyGS/tests/robot_tests/configs/episodes/dev_grid_drawer.yaml \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_retrieve_grid_drawer

``--repeats N`` runs the episode N times in one World (``World.next_episode``, seeds spec, spec+1, ...;
repeats after the first in ``repeat_<k>/``): every repeat must start from the same checked state
and retrieve the target as the first did, without building the World again.
"""

from __future__ import annotations

import time

from phygs_robot_tests.episodes import episode

from phygs.composite import CompositeSkills
from phygs.session import AgentFailed


def plan(session) -> None:
    """Hold the target in the gripper with the arm stowed."""
    skills = CompositeSkills(session)
    target, part = session.hints.target(), None
    if target.inside is not None:
        skills.open_drawer(session.hints.part(target.inside))
        target = session.hints.target()  # The open drawer moved it.
        part = session.hints.part(target.inside)
    skills.pick(target.center, target.size, name=target.name, part=part)
    episode.stow(session, skills.held)
    if not session.manipulation.holding:
        raise AgentFailed("The target dropped in the stow.")


def retrieved(check, report: dict, prefix: str = "") -> None:
    """The plan returned with the target held and the arm stowed (the evaluator's result)."""
    episode.agent_checks(check, report, prefix=prefix)
    evaluation = report["evaluation"]  # The evaluator's result from ground truth, after the stop.
    checks, measured = evaluation["checks"], evaluation["measured"]
    check.measure(f"{prefix}target_held", dict(height_m=measured["target_height_m"]), checks["held"], "not in the hand")
    check.measure(f"{prefix}stow_error_rad", measured["stow_error_rad"], checks["stowed"], "the arm is not stowed")
    check.measure(
        f"{prefix}evaluation", evaluation["outcome"], evaluation["outcome"] == "success", evaluation["reason"]
    )


def main() -> None:
    check, args, spec = episode.start(
        __doc__, "dev_grid_drawer", lambda flags: flags.add_argument("--repeats", type=int, default=1)
    )

    def body() -> None:
        if spec.task["goal"] != "retrieve":
            raise ValueError(f"{spec.id} is not a retrieve episode.")
        with episode.world(args, spec) as world:
            report = world.run(plan, name="check_retrieve.plan")
            retrieved(check, report)
            check.data["stopped"] = report["stopped"]
            runs = [dict(sim_seconds=report["sim_seconds"], wall_seconds=report["wall_seconds"])]
            for k in range(1, args.repeats):
                world.next_episode(args.output / f"repeat_{k}", seed=spec.seed + k)
                started = time.perf_counter()
                report = world.run(plan, name="check_retrieve.plan")
                # The repeat starts from the same checked state: the start check passed again.
                check.measure(f"repeat{k}_status", report["status"], report["status"] == "completed", report["reason"])
                retrieved(check, report, prefix=f"repeat{k}_")
                runs.append(
                    dict(
                        sim_seconds=report["sim_seconds"],
                        wall_seconds=report["wall_seconds"],
                        process_seconds=round(time.perf_counter() - started, 1),  # Reset, settle and run.
                        start_check=report["start_check"]["measured"],
                    )
                )
            check.data["runs"] = runs

    check.run(body)


if __name__ == "__main__":
    main()
