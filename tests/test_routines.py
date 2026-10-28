# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot tests' routines and the retrieve plan on a fake session: order, retries and geometry
(no Isaac Sim)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import SPOT
from scipy.spatial.transform import Rotation

from phygs.episodes.session import AgentFailed, SkillFailure, TargetHint, TrajectoryRefused
from phygs_robot_tests import routines
from phygs_robot_tests.episodes.check_retrieve import plan
from phygs_robot_tests.routines import PARAMS, arm, base
from phygs_robot_tests.routines import drawer as strokes
from phygs_robot_tests.routines.geometry import quaternion

DRAWER = SimpleNamespace(
    handle_center=(0.0, 2.56, 0.465),
    handle_axis=(1.0, 0.0, 0.0),
    handle_radius=0.009,
    direction=(0.0, -1.0, 0.0),  # The drawer opens toward -y.
    opening=0.0,
    limits=(0.0, 0.38),
)
# The open stance stands the far end of the grip range, less the walk's position tolerance, before
# the closed handle; a pull ends with the handle at the grip range's near end.
STANCE_M = PARAMS["grip_range_m"][1] - PARAMS["goal_tolerance"][0]
PULL_M = STANCE_M - PARAMS["grip_range_m"][0]
OPEN = (0.0, 2.56 - STANCE_M, math.pi / 2)  # Facing the handle.
STANCES = {
    "storage_drawer": {
        "close": (0.0, 1.38, math.pi / 2),
    },
    "table": {"approach": (-0.9, 0.0, 0.0), "reach": (-0.25, 0.0, 0.0)},
}


@dataclass(frozen=True)
class Feedback:
    status: str = "succeeded"
    reason: str = ""
    failure: SkillFailure | None = None
    attempts: int = 1
    progress: float = 1.0


REFUSED = Feedback("failed", "Arm planner failed: no plan", progress=0.0)  # A planned move refused before moving.
LINE_REFUSED = "refused"  # A line refused before moving (TrajectoryRefused).
STALLED = Feedback("failed", "End-effector tracking error exceeded tolerance", progress=0.5)


JOINTS = SPOT.arm.joints


@pytest.fixture(autouse=True)
def stances():
    """The routines' stances (normally configs/stances/<scene>.yaml, loaded by routines.use_scene)
    and Spot's facts (``session.robot``, loaded by routines.use_robot)."""
    routines.STANCES.clear()
    routines.STANCES.update(STANCES)
    routines.use_robot(SPOT.info())
    yield
    routines.STANCES.clear()


class FakeSession:
    """Clients return feedback as their command id; ``wait`` hands it back. Scripted results are
    consumed in order per kind (``approach``: planned moves, ``line``: end-effector lines,
    ``navigate_to``); the rest succeed. The tool pose follows the moves that succeed."""

    def __init__(self, inside="storage_drawer", heading=math.pi / 2, **results):
        self.position = (0.0, 0.0)
        self.calls, self.results, self.heading = [], {k: list(v) for k, v in results.items()}, heading
        self.tool = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        target = TargetHint("target", (0.0, 2.4, 0.1), (0.04, 0.04, 0.12), inside, None if inside else "table")
        self.hints = SimpleNamespace(
            target=lambda: target,
            part=lambda name: DRAWER,
        )
        self.navigation = SimpleNamespace(navigate_to=self._navigate)
        self.manipulation = SimpleNamespace(
            holding=False,
            episode_id=1,
            stow=lambda held=None: self._call("stow", held),
            look_at=lambda target, **view: self._call("look_at"),
            open_gripper=lambda: self._gripper(False),
            close_gripper=lambda: self._gripper(True),
            move_to_pose=self._move_to_pose,
            execute_trajectory=self._line,
            tool_pose=lambda tool="hand": tuple(self.tool),
            move_joints=lambda targets, duration: self._call("move_joints"),
        )

    def _call(self, name, *details, kind=None):
        self.calls.append((name, *details))
        results = self.results.get(kind or name)
        return results.pop(0) if results else Feedback()

    timeouts: list = []

    def _navigate(self, x, y, heading, timeout=None, **tolerance):
        self.timeouts = [*self.timeouts, timeout]
        self.heading, self.position = heading, (x, y)
        return self._call("navigate_to", (x, y, heading))

    def _gripper(self, close):
        self.manipulation.holding = close and self._near
        return self._call("close_gripper" if close else "open_gripper")

    _near = False

    def _move_to_pose(self, pose, *, held=None, tool="hand"):
        feedback = self._call("approach", tuple(np.round(pose, 4)), held, tool)
        if feedback.status == "succeeded":
            self.tool = np.asarray(pose, dtype=float)
        self._near = False
        return feedback

    def _line(self, line):
        start, end = np.asarray(line.poses[0]), np.asarray(line.poses[-1])
        motion = end[:3] - start[:3]
        distance = float(np.linalg.norm(motion))
        feedback = self._call("line", tuple(np.round(motion / distance, 3)), round(distance, 3))
        if feedback == LINE_REFUSED:
            raise TrajectoryRefused("out_of_reach", "The trajectory leaves the arm's reach.")
        if feedback.status == "succeeded":
            self.tool, self._near = end, True
        return feedback

    def wait(self, feedback):
        return feedback

    def sleep(self, seconds):
        self.calls.append(("sleep", seconds))

    def observe(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        x, y = self.position
        pose = SimpleNamespace(matrix=np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
        joints = SimpleNamespace(names=JOINTS, positions=np.zeros(len(JOINTS)))  # Not stowed.
        return SimpleNamespace(localization=SimpleNamespace(valid=True, pose=pose), joints=joints)

    def names(self):
        return [call[0] for call in self.calls]

    def lines(self):
        return [call for call in self.calls if call[0] == "line"]


def test_the_retrieve_plan_opens_the_drawer_then_reaches_in_picks_and_stows():
    session = FakeSession()
    plan(session)
    releases = ["line"] * len(PARAMS["handle_release"])
    assert session.names() == [
        "stow",
        "navigate_to", "sleep",  # The open stance; the legs settle.
        "look_at",  # The handle.
        "open_gripper", "approach", "line", "close_gripper", "line", "open_gripper", *releases, "sleep",
        "stow",
        "navigate_to", "sleep",  # The reach-in stance.
        "look_at", "open_gripper", "approach", "line", "close_gripper", "line", "sleep",  # Pick: in, lift.
        "stow",
    ]  # fmt: skip
    open_walk, reach_walk = (call[1] for call in session.calls if call[0] == "navigate_to")
    assert open_walk == pytest.approx(OPEN)
    assert reach_walk == pytest.approx((0.0, 2.4 - PARAMS["reach_in_ahead_m"], math.pi / 2))  # The target ahead.
    into, slide, *release = session.lines()[: 2 + len(PARAMS["handle_release"])]
    assert into[2] == pytest.approx(PARAMS["handle_approach_m"][0])  # Straight onto the handle.
    assert slide[1] == (0.0, -1.0, 0.0) and slide[2] == pytest.approx(PULL_M)  # Until the grip range's near end.
    assert [call[2] for call in release] == [distance for _, distance in PARAMS["handle_release"]]
    approach = np.asarray(into[1])  # Into the part.
    for (move, _), call in zip(PARAMS["handle_release"], release, strict=True):
        axis = np.asarray(call[1])
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-3)
        if move == "gripper_back":
            assert axis == pytest.approx(-approach, abs=1e-3)  # Straight back out along the gripper axis.
        if move == "wrist_down":
            assert abs(axis @ approach) < 1e-3
    pick_in, lift = session.lines()[-2:]
    distance = PARAMS["approach_m"]
    assert pick_in[1:] == ((0.0, 0.0, -1.0), pytest.approx(distance))  # Straight down first.
    assert lift[1:] == ((0.0, 0.0, 1.0), pytest.approx(distance))
    held = session.calls[-1][1]  # The stow carries the picked object.
    assert held.name == "target" and held.size == (0.04, 0.04, 0.12)


def test_the_pick_uses_the_target_where_the_open_drawer_moved_it():
    session = FakeSession()
    hints = [TargetHint("target", (0.0, y, 0.1), (0.04, 0.04, 0.12), "storage_drawer", None) for y in (2.4, 2.1)]
    session.hints.target = lambda: hints.pop(0) if len(hints) > 1 else hints[0]  # Read again: opened.
    plan(session)
    pregrasp = [call[1] for call in session.calls if call[0] == "approach"][-1]
    assert abs(pregrasp[1] - 2.1) < 0.05  # Above the moved target, not where it was before.


def test_the_open_stance_is_the_same_rule_for_every_drawer():
    session = FakeSession()
    assert base.open_stance(session, "storage_drawer") == pytest.approx(OPEN)
    # A half-open drawer facing +x: the stance is measured from where its handle is when closed.
    turned = SimpleNamespace(
        **{**vars(DRAWER), "handle_center": (3.2, 1.0, 0.3), "direction": (1.0, 0.0, 0.0), "opening": 0.2}
    )
    session.hints.part = lambda name: turned
    x, y, heading = base.open_stance(session, "drawer")
    assert (x, y) == pytest.approx((3.0 + STANCE_M, 1.0)) and (math.cos(heading), math.sin(heading)) == pytest.approx(
        (-1.0, 0.0)
    )  # Facing -x.


def test_a_drawer_place_spot_keeps_its_distance_from_the_carcass_whatever_the_opening():
    """The opened jaw reaches toward the carcass: at the floor's middle a 0.262 m opening left the
    box 0.127 m from it and the jaw caught its edge (2026-10-25). The spot moves out to the
    clearance, but no closer to the drawer front than the margin."""
    session = FakeSession()
    size = (0.04, 0.04, 0.12)
    clearance, margin = PARAMS["place_carcass_clearance_m"], PARAMS["place_front_margin_m"]
    for opening in (0.262, 0.284, 0.38):  # Wide enough: the clearance, or the middle if that is farther.
        floor = (0.0, 2.6 - opening / 2, 0.0125)  # The exposed floor's middle; the carcass front is at y 2.6.
        session.hints.part = lambda name, opening=opening, floor=floor: SimpleNamespace(
            **{**vars(DRAWER), "opening": opening, "floor_center": floor}
        )
        x, y, z = strokes.place_spot(session, "storage_drawer", size)
        assert (x, 2.6 - y, z) == pytest.approx((0.0, max(clearance, opening / 2), 0.0125))
    narrow = 0.2  # Too narrow for the clearance: as far out as the front margin allows.
    session.hints.part = lambda name: SimpleNamespace(
        **{**vars(DRAWER), "opening": narrow, "floor_center": (0.0, 2.6 - narrow / 2, 0.0125)}
    )
    assert 2.6 - strokes.place_spot(session, "storage_drawer", size)[1] == pytest.approx(narrow - 0.02 - margin)


def test_a_target_on_a_table_is_reached_through_the_approach_stance():
    session = FakeSession(inside=None, heading=0.0)
    plan(session)
    stances = [call[1] for call in session.calls if call[0] == "navigate_to"]
    assert stances == [STANCES["table"]["approach"], STANCES["table"]["reach"]]
    assert session.names().count("approach") == 1  # No drawer to open: only the pick's planned move.


def test_refused_options_move_on_in_order():
    # The first handle approach (both bar directions at the first distance) is refused; the pick's
    # straight-down approach is refused at every yaw, so it moves on to the first tilted approach,
    # given in the robot's frame.
    yaws = len(PARAMS["grasp_yaws_deg"])
    session = FakeSession(approach=[REFUSED, REFUSED, Feedback(), *[REFUSED] * yaws])
    plan(session)
    handle_in = session.lines()[0]
    assert handle_in[2] == pytest.approx(PARAMS["handle_approach_m"][1])  # 0.06 m: the next distance.
    pick_in = session.lines()[-2]
    forward, side, down = PARAMS["pick_approaches"][1]  # In the robot's frame; the robot heads +y.
    assert pick_in[1] == pytest.approx((-side, forward, down), abs=1e-3)


def test_when_every_pitch_is_refused_the_plan_walks_to_the_stance_again():
    refusals = [REFUSED] * (2 * len(PARAMS["handle_approach_m"]) * len(PARAMS["handle_pitch_deg"]))
    session = FakeSession(approach=refusals)
    plan(session)
    names = session.names()
    assert names.count("navigate_to") == 3  # Open stance, again after the refusals, reach-in stance.
    with pytest.raises(AgentFailed, match="every handle pitch"):
        plan(FakeSession(approach=refusals * PARAMS["stance_tries"]))


def test_a_refused_release_move_is_tried_at_half_then_ends_the_release():
    moves = PARAMS["handle_release"]
    assert len(moves) >= 2
    # In and the slide succeed; the first release move is refused at full and at half length.
    session = FakeSession(line=[Feedback(), Feedback(), LINE_REFUSED, LINE_REFUSED])
    plan(session)
    lines = session.lines()
    assert [call[2] for call in lines[2:4]] == pytest.approx([moves[0][1], moves[0][1] / 2])
    after = session.calls[session.calls.index(lines[3]) + 1]
    assert after[0] == "sleep"  # The later release moves were skipped.


def test_a_timed_out_walk_is_issued_again_once():
    timed_out = Feedback("timed_out", "Navigation time budget exhausted")
    session = FakeSession(navigate_to=[timed_out])
    plan(session)
    stances = [call for call in session.calls if call[0] == "navigate_to"]
    assert stances[0][1] == stances[1][1] == pytest.approx(OPEN)
    assert session.timeouts[0] == PARAMS["walk_timeout_s"]
    with pytest.raises(AgentFailed, match="timed_out"):
        plan(FakeSession(navigate_to=[timed_out] * PARAMS["walk_tries"]))
    with pytest.raises(AgentFailed, match="blocked"):
        plan(FakeSession(navigate_to=[Feedback("blocked", "Goal unreachable")]))


def test_a_failure_while_moving_ends_the_plan():
    with pytest.raises(AgentFailed, match="slide"):
        plan(FakeSession(line=[Feedback(), STALLED]))  # In, then the slide stalls.
    before_lift = 2 + len(PARAMS["handle_release"]) + 1  # Handle in, slide, releases, pick in.
    with pytest.raises(AgentFailed, match="pick lift"):
        plan(FakeSession(line=[Feedback()] * before_lift + [STALLED]))


@pytest.mark.parametrize("robot", ["spot", "ridgeback_franka"])
def test_the_pick_grasp_pose_puts_the_grasp_frame_on_the_point_along_the_approach(robot):
    from phygs.robots.description import load_robot

    routines.use_robot(load_robot(robot).info())
    point = np.array([0.5, 0.2, 0.3])
    for approach in (arm.DOWN, np.array([0.5, 0.0, -0.866])):
        approach = approach / np.linalg.norm(approach)
        wrist = arm.grasp_wrist_pose(point, approach, math.radians(90.0))
        assert arm.gripper_axes(wrist)[:, 0] == pytest.approx(approach, abs=1e-6)  # The gripper's approach.
        assert (wrist @ arm.tool("grasp"))[:3, 3] == pytest.approx(point, abs=1e-9)  # The grasp point reaches it.
    if robot == "spot":  # Spot's gripper axes are its wrist's: the wrist x is the approach.
        assert wrist[:3, 0] == pytest.approx(approach, abs=1e-6)
    else:  # The Franka hand approaches along its z.
        assert wrist[:3, 2] == pytest.approx(approach, abs=1e-6)


@pytest.mark.parametrize("side", [1.0, -1.0])
@pytest.mark.parametrize("pitch", [0.0, 15.0, 30.0, 45.0])
def test_the_handle_grasp_holds_the_bar_along_the_jaws_pitched_down_into_the_drawer(side, pitch):
    drawer = SimpleNamespace(**{**vars(DRAWER), "direction": (0.6, -0.8, 0.0), "handle_axis": (0.8, 0.6, 0.0)})
    pose = strokes.handle_grasp_pose(drawer, side, pitch)
    rotation = pose[:3, :3]
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12) and np.linalg.det(rotation) == pytest.approx(
        1.0
    )
    assert rotation[:, 1] == pytest.approx(np.multiply(drawer.handle_axis, side), abs=1e-12)  # The bar along y.
    into, down = -np.asarray(drawer.direction), math.radians(pitch)
    expected_axis = math.cos(down) * into + [0.0, 0.0, -math.sin(down)]  # Into the drawer, pitched down.
    assert rotation[:, 0] == pytest.approx(expected_axis, abs=1e-12)
    wrist = arm.wrist_for("handle", pose)
    bar = (np.linalg.inv(wrist) @ [*drawer.handle_center, 1.0])[:3]  # The bar's centre in the wrist frame.
    # Spot: in the jaws at x 0.22 m, resting on the fixed jaw (0.029 m below the wrist axis, plus the radius).
    assert bar == pytest.approx([0.22, 0.0, -0.029 + drawer.handle_radius], abs=1e-12)
    assert abs(np.dot(quaternion(rotation), Rotation.from_matrix(rotation).as_quat())) == pytest.approx(1.0)


def test_the_routines_load_their_stances_by_scene():
    """``use_scene`` reads configs/stances/<scene>.yaml (the routines' own data); a scene without a
    file has no stances, and asking for one ends the plan."""
    routines.use_scene("house")
    assert routines.STANCES["kitchen_island_drawer"]["close"] == [11.8, 5.46, 1.5708]
    routines.use_scene("no_such_scene")
    assert routines.STANCES == {}
    with pytest.raises(AgentFailed, match="No stance"):
        base.go_to(FakeSession(), "kitchen_island_drawer", "close")


def test_reaching_into_an_open_drawer_stops_clear_of_it_first():
    """The reach-in stance puts the point reach_in_ahead_m ahead; when that puts the body over the
    open drawer (its handle nearer than reach_in_clear_m), a stance with the handle that far ahead
    comes first, on the same line (navigation's fine approach makes the last bit)."""
    session = FakeSession()
    opened = SimpleNamespace(**{**vars(DRAWER), "handle_center": (0.0, 2.21, 0.465), "opening": 0.35})
    session.hints.part = lambda name: opened
    clear, stance = base.reach_in_stances(session, "storage_drawer", (0.0, 2.5, 0.44))
    assert stance == pytest.approx((0.0, 2.5 - PARAMS["reach_in_ahead_m"], math.pi / 2))
    assert clear == pytest.approx((0.0, 2.21 - PARAMS["reach_in_clear_m"], math.pi / 2))
    assert len(base.reach_in_stances(session, "storage_drawer", (0.0, 2.3, 0.44))) == 1  # Clear of it already.
    with pytest.raises(ValueError, match="point"):
        base.go_to(session, "storage_drawer", "reach_in")
