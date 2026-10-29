# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walking: to a stance within the routines' tolerance, then standing while the legs settle."""

from __future__ import annotations

import math

import numpy as np
from phygs_robot_tests.routines import PARAMS, STANCES

from phygs.session import AgentFailed


def go_to(session, place: str, use: str, point=None) -> None:
    """Walk to the stance for ``use`` at ``place``, then stand while the legs settle; the arm must
    be stowed. ``open`` and ``reach_in`` are the same rule for every drawer (:func:`open_stance`,
    :func:`reach_in_stances`; ``reach_in`` needs the ``point`` [m] the arm reaches in to); other
    uses are the routines' stances for the scene (:data:`routines.STANCES`), through an
    ``approach`` stance when going to ``reach`` and one is given."""
    if use == "open":
        walk(session, open_stance(session, place))
    elif use == "reach_in":
        if point is None:
            raise ValueError("reach_in needs the point the arm reaches in to.")
        for stance in reach_in_stances(session, place, point):
            walk(session, stance)
    else:
        uses = STANCES.get(place, {})
        if use not in uses:
            raise AgentFailed(f"No stance for {place!r} {use!r}; the routines have {sorted(STANCES)}.")
        if use == "reach" and "approach" in uses:
            walk(session, tuple(uses["approach"]))
        walk(session, tuple(uses[use]))
    session.sleep(PARAMS["settle_seconds"])


def open_stance(session, drawer: str) -> tuple[float, float, float]:
    """Where to stand to open a drawer (x, y, heading) [m, m, rad]: facing it, with its closed
    handle at the far end of ``grip_range_m`` less the walk's position tolerance ahead of the
    base centre, so wherever the base stops within that tolerance the handle is in reach."""
    part = session.hints.part(drawer)
    out = outward(part)
    closed = np.asarray(part.handle_center, dtype=float)[:2] - out * part.opening
    x, y = closed + out * (PARAMS["grip_range_m"][1] - PARAMS["goal_tolerance"][0])
    return float(x), float(y), math.atan2(-out[1], -out[0])


def reach_in_stances(session, drawer: str, point) -> list[tuple[float, float, float]]:
    """Where to stand (x, y, heading) [m, m, rad] to reach into an open drawer at ``point`` [m]:
    facing the drawer with the point ``reach_in_ahead_m`` ahead of the base centre. When that
    puts the body over the open drawer, a clear stance comes first, with the drawer's handle
    ``reach_in_clear_m`` ahead: navigation walks there on its map and finishes the last bit with
    its fine approach."""
    part = session.hints.part(drawer)
    out = outward(part)
    heading = math.atan2(-out[1], -out[0])
    point = np.asarray(point, dtype=float)[:2]
    stance = point + out * PARAMS["reach_in_ahead_m"]
    handle_ahead = float(np.dot(np.asarray(part.handle_center, dtype=float)[:2] - stance, -out))
    stances = [(float(stance[0]), float(stance[1]), heading)]
    if handle_ahead < PARAMS["reach_in_clear_m"]:
        clear = stance + out * (PARAMS["reach_in_clear_m"] - handle_ahead)
        stances.insert(0, (float(clear[0]), float(clear[1]), heading))
    return stances


def walk(session, stance: tuple[float, float, float]) -> None:
    """navigate_to a stance (x, y, heading) [m, m, rad] within ``goal_tolerance``.

    Each attempt has its own time limit (``walk_timeout_s``); a timed-out attempt is issued again,
    up to ``walk_tries`` attempts (near a goal the follower can stand still just outside the
    tolerance, 2026-10-15). Any other failure ends the plan.
    """
    position_tolerance, heading_tolerance = PARAMS["goal_tolerance"]
    for _ in range(PARAMS["walk_tries"]):
        feedback = session.wait(
            session.navigation.navigate_to(
                *stance,
                position_tolerance=position_tolerance,
                heading_tolerance=heading_tolerance,
                timeout=PARAMS["walk_timeout_s"],
            )
        )
        if feedback.status == "succeeded":
            return
        if feedback.status != "timed_out":
            break
        print(f"routines: navigate_to {stance} timed out ({feedback.reason}); trying again", flush=True)
    raise AgentFailed(f"navigate_to {stance}: {feedback.status}: {feedback.reason}")


def pose(session) -> np.ndarray:
    """The base's world pose (4x4) from the localization the session observes."""
    localization = session.observe().localization
    if localization is None or not localization.valid:
        raise AgentFailed("The routines need the base pose (gt_localization).")
    return np.asarray(localization.pose.matrix.tolist())


def heading(session) -> float:
    """The base's heading [rad]."""
    matrix = pose(session)
    return math.atan2(matrix[1, 0], matrix[0, 0])


def outward(part) -> np.ndarray:
    """The horizontal unit vector a part opens toward."""
    direction = np.asarray(part.direction, dtype=float)[:2]
    return direction / np.linalg.norm(direction)
