# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drawer strokes: open the drawer the target starts in by its handle from one stance, as far as
the grip range allows (until the handle is 0.80 m ahead), and push it shut from another.

The opening after the pull is read through the part hint the spec grants (the plan runs on, so
ground truth cannot be read then); the closed drawer is judged with ground truth.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_drawer_stroke.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_drawer_stroke
"""

from __future__ import annotations

from phygs_robot_tests.episodes import episode
from phygs_robot_tests.routines import arm, base, drawer


def plan(session, seen: dict) -> None:
    """Open, then close, the target's drawer; ``seen`` keeps the drawer, the planned pull and the
    opening after it [m]."""
    name = seen["drawer"] = session.hints.target().inside
    arm.stow(session)
    base.go_to(session, name, "open")
    arm.look_at(session, session.hints.part(name).handle_center, "handle")
    seen["planned_m"] = drawer.open_drawer(session, name)
    seen["opened_m"] = session.hints.part(name).opening
    arm.stow(session)
    base.go_to(session, name, "close")
    drawer.close_drawer(session, name)
    arm.stow(session)


def main() -> None:
    check, args, spec = episode.start(__doc__, "dev_grid_drawer")
    settings = episode.SETTINGS

    def body() -> None:
        seen = {}
        with episode.world(args, spec) as world:
            report = world.run(lambda session: plan(session, seen), name="check_drawer_stroke.plan")
            episode.agent_checks(check, report)
            planned, opened = seen["planned_m"], seen["opened_m"]
            check.measure("planned_m", round(planned, 4), planned >= settings["stroke_min_opening_m"], "pull too short")
            least = planned - settings["opening_tolerance_m"]
            check.measure("opened_m", round(opened, 4), opened >= least, f"opened less than {least} m")
            closed = world.truth().part(seen["drawer"]).opening
            check.measure("closed_m", round(closed, 4), closed <= settings["closed_max_m"], "the drawer did not close")

    check.run(body)


if __name__ == "__main__":
    main()
