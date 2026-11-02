# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The composite skills' tuning: what they try, in which order, and the distances they keep.

These are tuning, not robot facts: measured for Spot in the grid and house scenes (the dates say
when), the same for every scene. Robot facts (tool frames, arm joints, stow postures) come from
``session.robot``. Another robot passes its own :class:`Tuning`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tuning:
    """Distances [m], angles [deg or rad, as named], durations [s] and option lists, tried in order."""

    # navigate_to position [m] and heading [rad] tolerance at a stance: navigate_to's default 0.04 m.
    # A narrow band keeps the open stance useful (with 0.08 m the base stopped where a drawer opened
    # only 0.20 m, 2026-10-16); 0.03 m took up to 8 fine-approach moves at a stance (2026-10-27).
    goal_tolerance: tuple[float, float] = (0.04, 0.05)
    settle_seconds: float = 1.0  # Standing still after a drawer stroke, so the drawer comes to rest [s].
    # The handle of a drawer the gripper holds stays within this range ahead of the base centre [m].
    # Opening stands with the handle at the far end less the walk's position tolerance (1.09 m; the
    # base stops within 1.05-1.13 m) and pulls until the handle is at the near end, measured from
    # where the base stopped. 1.13 m: the 30 deg pitched grasp's reach for the lowest handles
    # (output_check/20261013-1410_arm_reach; a base stopped past it refused every grasp, 2026-10-16).
    # 0.80 m: the wrist ends ~0.30 m behind the grip point, ~5 cm in front of the body (0.45 m); at
    # 0.71-0.75 m it ended under the chest and no stow could start (2026-10-15). Closing centres the
    # hand's push in the range.
    grip_range_m: tuple[float, float] = (0.80, 1.13)
    # Reaching for a point (an object, a place spot): stand facing it this far ahead of the base centre
    # [m] (0.94-0.95 in the grid and house drawers, where the picks and places work). In an open
    # drawer that puts the body over the drawer, which a sensed costmap marks occupied, so the walk
    # first stops with the drawer's handle reach_in_clear_m ahead [m]: the padded outline's front
    # (0.66 m) plus 0.05 m plus a 5 cm costmap cell. From there navigation's fine approach (Nav2
    # occupied_goal_reach, 0.35 m) makes the last bit.
    reach_ahead_m: float = 0.95
    reach_in_clear_m: float = 0.80
    # A point within this of the base's heading [deg] is reached keeping the heading: square to the
    # open drawer the robot stands at. Along the line of sight, 16 deg off square, a pick in the house's
    # bottom drawer closed on nothing twice (pointing oracle, 2026-11-01).
    keep_heading_deg: float = 45.0
    # When navigate_to refuses the reach stance as unreachable (the robot does not fit there: facing a
    # table's corner from the drawer, drawer_place 2026-11-01), the stance turns about the point by
    # the next of these [deg]. Square to a drawer it does not turn.
    reach_turns_deg: tuple[float, ...] = (0.0, 30.0, -30.0, 60.0, -60.0)
    # When every turn is refused, the turns are tried again this much farther from the point [m],
    # within the grip range. Before a kitchen counter, a bottle 0.24 m behind its front put the padded
    # outline's front (0.66 m ahead of the base) on the costmap's lethal cells from 0.95 m at every
    # turn; from 1.00 m it fit (Qwen3.8, house_drawer, 2026-11-01). In a drawer it does not step back.
    reach_back_m: tuple[float, ...] = (0.0, 0.10)
    limit_margin_m: float = 0.01  # Stop this far [m] short of a slide's end.
    # Stand again when no handle pitch works from where the base stopped: an opening one walk
    # tolerance nearer each time (the base had stopped at the far edge, 2026-10-30).
    stance_tries: int = 2
    # Handle grasp pitches [deg] (gripper axis down about the bar), tried in order while the approach
    # or the slide move is refused before moving; approach distances [m] along the gripper axis.
    handle_pitch_deg: tuple[float, ...] = (30.0, 45.0, 15.0, 0.0)
    handle_approach_m: tuple[float, ...] = (0.08, 0.06, 0.04)
    # After letting go of the handle, the retreat: (move, distance [m]); wrist_down (wrist -z),
    # gripper_back (wrist -x), up (world z), forward (the gripper axis made horizontal). Off the bar
    # and back until the jaw tip (0.016 m past the bar's axis) and the planner's padding are clear;
    # the stow plans from there (2026-10-15: the gripper's surface-fitted planner spheres).
    handle_release: tuple[tuple[str, float], ...] = (("wrist_down", 0.02), ("gripper_back", 0.04))
    # Closing pushes with the closed gripper, stopped this far short of the bar [m], then backs off.
    push_back_m: float = 0.04
    push_release: tuple[tuple[str, float], ...] = (("gripper_back", 0.03), ("up", 0.10))
    # Pick approaches tried in order: None is straight down; vectors are in the robot's frame (x
    # forward, z up), tilted away from the robot.
    pick_approaches: tuple[tuple[float, float, float] | None, ...] = (
        None,
        (0.5, 0.0, -0.866),
        (0.7071, 0.0, -0.7071),
    )
    grasp_yaws_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 180.0)  # Rolls about the approach, tried in order.
    approach_m: float = 0.19  # Straight in to a grasp or a place, and back out [m].
    hold_seconds: float = 1.0  # Hold after the pick's lift before stowing [s].
    place_clearance_m: float = 0.04  # Release the object this far above the surface [m].
    place_wrist_height_m: float = 0.5  # place_upright turns the wrist top-down this far above the spot [m].
    # A drawer place spot this far from the carcass front [m]: the opened moving jaw reaches toward the
    # carcass and caught its front edge on the way back up when the box stood in the middle of a drawer
    # opened 0.262 m (0.127 m from the carcass, 33 N, blocked; 0.138 m at 0.284 m brushed it at 2 N,
    # 2026-10-25). The box keeps place_front_margin_m [m] from the drawer front: 0.05 m from it the
    # planned move above the spot with the box held was refused, 0.075 m and more worked (drawer_place,
    # 2026-11-01). An opening narrower than 0.27 m moves the spot toward the carcass instead.
    place_carcass_clearance_m: float = 0.17
    place_front_margin_m: float = 0.08
    line_speed_m_s: float = 0.1  # Straight end-effector lines [m/s].
    stow_tolerance_rad: float = 0.2  # Within this of the stow (every arm joint), the arm counts as stowed [rad].
    # look_at views (distance [m], elevation [deg]), tried in order: a drawer's handle (low handles,
    # the house's bottom drawer, need the later, shallower views) and a place spot, e.g. an open
    # drawer's floor. An object is looked at from look_at's own views.
    look_views: dict[str, tuple[tuple[float, float], ...]] = field(
        default_factory=lambda: {
            "handle": ((0.6, 60.0), (0.5, 45.0), (0.45, 45.0), (0.6, 30.0)),
            "place": ((0.6, 75.0), (0.7, 60.0), (0.6, 60.0), (0.7, 45.0)),
        }
    )


TUNING = Tuning()
