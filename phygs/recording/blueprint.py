# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The viewer's layout for an episode's recording (:mod:`phygs.recording.log`): the world in 3D with
Nav2's route; beside it the agent's calls and own records, what the agent saw, the images of its own
records, Nav2's costmap, the last arm plan and the joint targets and efforts (tabs) over the events
and contacts; plots under it (joints, contact forces, base command, real-time factor). Sent with the
recording, so ``rerun episode.rrd`` opens in this layout."""

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
                    rrb.TimeSeriesView(name="Contact forces [N]", origin="/truth/contact_forces"),
                    rrb.TimeSeriesView(
                        name="Base command, holding", contents=["/control/base_command", "/estimate/holding"]
                    ),
                    rrb.TimeSeriesView(name="Real-time factor", origin="/perf/rtf"),
                ),
                row_shares=[3, 1],
            ),
            rrb.Vertical(
                rrb.Tabs(
                    rrb.TextLogView(name="Agent calls and records", origin="/agent"),
                    rrb.Spatial2DView(name="What the agent saw", origin="/agent_view/images"),
                    rrb.Spatial2DView(name="Agent's record images", origin="/agent/log_images"),
                    rrb.Spatial2DView(name="Nav2 costmap", origin="/nav/costmap"),
                    rrb.Spatial3DView(name="Arm plan (body frame)", origin="/planner/plan"),
                    rrb.TimeSeriesView(
                        name="Joint targets, efforts", contents=["/control/joint_targets", f"{entity}/joint_efforts"]
                    ),
                ),
                rrb.TextLogView(name="Events and contacts", contents=["/events/**", "/truth/contact_events"]),
                row_shares=[3, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="sim_time"),
        collapse_panels=False,
    )
