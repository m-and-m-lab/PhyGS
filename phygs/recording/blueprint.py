# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The viewer's layout for an episode's recording (:mod:`phygs.recording.log`): the world in 3D with
Nav2's route; beside it the agent's calls, Nav2's costmap and the last arm plan (tabs) over the events
and contacts; plots under it. Sent with the recording, so ``rerun episode.rrd`` opens in this layout."""

from __future__ import annotations


def episode_blueprint(robot: str, root: str):
    """The layout for a recording of ``robot`` (its URDF name: the ``/truth/<robot>`` entities) whose
    root link is ``root``; the 3D view follows the root link's geometry."""
    import rerun as rr
    import rerun.blueprint as rrb

    entity = f"/truth/{rr.escape_entity_path_part(robot)}"
    follow = rrb.EyeControls3D(tracking_entity=f"{entity}/collision_geometries/{rr.escape_entity_path_part(root)}")
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="World", origin="/", contents=["/truth/**", "/nav/route"], eye_controls=follow),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Joints [rad, m]", origin=f"{entity}/joint_positions"),
                    rrb.TimeSeriesView(name="Base command, holding", contents=["/control/**", "/estimate/**"]),
                    rrb.TimeSeriesView(name="Real-time factor", origin="/perf/rtf"),
                ),
                row_shares=[3, 1],
            ),
            rrb.Vertical(
                rrb.Tabs(
                    rrb.TextLogView(name="Agent calls", origin="/agent"),
                    rrb.Spatial2DView(name="Nav2 costmap", origin="/nav/costmap"),
                    rrb.Spatial3DView(name="Arm plan (body frame)", origin="/planner/plan"),
                ),
                rrb.TextLogView(name="Events and contacts", contents=["/events/**", "/truth/contacts"]),
                row_shares=[3, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="sim_time"),
        collapse_panels=False,
    )
