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
    limit_margin_m: float = 0.01  # Stop this far [m] short of a slide's end.
    # Stand again when the handle approach or the slide move is refused before moving from where the
    # base stopped: an opening one walk tolerance nearer each time (the base had stopped at the far
    # edge, 2026-10-30).
    stance_tries: int = 2
    # The handle grasp pitches the gripper axis down about the bar along the line from the arm's
    # shoulder to the handle, from where the base stopped, at most this [deg] (level for a handle
    # above the shoulder). Spot's arm reaches its highest handles (0.67-0.80 m) along the line from 9
    # of 9 base poses at the stance, at a fixed 30 deg from 3-7 (the reach check, offline). At 36.7 deg
    # on a 0.155 m handle the jaw stopped on the drawer front; at 30 deg every low handle opened
    # (2026-11-03).
    handle_pitch_max_deg: float = 30.0
    handle_approach_m: float = 0.08  # The planned move ends this far before the handle [m]; then straight in.
    # A grip on a handle bar closes the gripper at least this fraction of the way from open to closed.
    # Spot on a 2 cm bar: 0.88 (-0.21 rad). With the jaw stopped on the drawer front: 0.05 (-1.33 rad),
    # and holding was true there too (2026-11-03).
    handle_grip_closed: float = 0.5
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
    # look_at's sweep: sides visited in order, each a list of views (distance [m], elevation [deg],
    # azimuth [deg] about the vertical through the target, counterclockwise seen from above) that the
    # session's look_at tries in order. One ring 0.6 m out at -45, -25, 0, 25 and 45 deg around, 45 deg
    # up, higher where that is out of reach. Of 40 views (0.45/0.6 m, 30-75 deg, -45 to 45 deg) on five
    # stances, 2026-11-04, depth planning: with the target 0.96-1.28 m from the base (the house's
    # bottom drawer from its open stance, grid_drawer, grid_table) the 45 deg ring was reached in 14 of
    # 15 cases (-45 deg in the house not) and saw 3-4 of the house bottle's 13 sample points, 9-11 in the
    # grid scenes; 30 deg views saw none in the drawers. With a table target 0.80 m away 0, 25 and 45
    # deg at 45 deg up were out of reach (the camera over the shoulder); 0 and 25 deg were reached at
    # 60 deg, 45 deg at 75 deg, each seeing 10-11 points. At 0.95 m (the pick's reach stance) the ring
    # was reached whole.
    look_sweep: tuple[tuple[tuple[float, float, float], ...], ...] = tuple(
        ((0.6, 45.0, azimuth), (0.6, 60.0, azimuth), (0.6, 75.0, azimuth))
        for azimuth in (-45.0, -25.0, 0.0, 25.0, 45.0)
    )
    look_views: dict[str, tuple[tuple[float, float], ...]] = field(
        default_factory=lambda: {
            "handle": ((0.6, 60.0), (0.5, 45.0), (0.45, 45.0), (0.6, 30.0)),
            "place": ((0.6, 75.0), (0.7, 60.0), (0.6, 60.0), (0.7, 45.0)),
        }
    )


TUNING = Tuning()
