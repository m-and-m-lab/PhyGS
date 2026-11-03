# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The composite skills (pick, place, open_drawer, close_drawer) and the robot tests' retrieve plan
on a fake session: order, retries, the options and measurements they report, and geometry (no Isaac
Sim)."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import SPOT
from phygs_robot_tests.episodes.check_retrieve import plan
from scipy.spatial.transform import Rotation

from phygs.composite import TUNING, CompositeSkills
from phygs.composite.geometry import quaternion
from phygs.composite.skills import DOWN
from phygs.session import AgentFailed, ArticulatedPart, SkillFailure, TargetHint, TrajectoryRefused

DRAWER = ArticulatedPart(
    handle_center=(0.0, 2.56, 0.465),
    handle_axis=(1.0, 0.0, 0.0),
    handle_length=0.1,
    handle_radius=0.009,
    direction=(0.0, -1.0, 0.0),  # The drawer opens toward -y.
    opening=0.0,
    limits=(0.0, 0.38),
)
# The open stance stands the far end of the grip range, less the walk's position tolerance, before
# the handle; a pull ends with the handle at the grip range's near end.
STANCE_M = TUNING.grip_range_m[1] - TUNING.goal_tolerance[0]
PULL_M = STANCE_M - TUNING.grip_range_m[0]
OPEN = (0.0, 2.56 - STANCE_M, math.pi / 2)  # Facing the handle.
BODY_Z = 0.541  # Spot's standing body height [m] (2026-11-02).


@dataclass(frozen=True)
class Feedback:
    status: str = "succeeded"
    reason: str = ""
    failure: SkillFailure | None = None
    attempts: int = 1
    progress: float = 1.0
    gripper_aperture: float | None = None


REFUSED = Feedback("failed", "Arm planner failed: no plan", progress=0.0)  # A planned move refused before moving.
LINE_REFUSED = "refused"  # A line refused before moving (TrajectoryRefused).
STALLED = Feedback("failed", "End-effector tracking error exceeded tolerance", progress=0.5)
JOINTS = SPOT.arm.joints
ROBOT = SPOT.info()
ON_BAR = -0.21  # Spot's gripper closed on a 2 cm handle bar [rad] (2026-11-03).


class FakeSession:
    """Clients return feedback as their command id; ``wait`` hands it back. Scripted results are
    consumed in order per kind (``approach``: planned moves, ``line``: end-effector lines,
    ``navigate_to``, ``stow``, ``look_at``, ``close_gripper``); the rest succeed. The tool pose follows
    the moves that succeed; the arm joints are at the stow posture after a stow and out after any other
    arm move. A close at an object stops at ``ON_BAR``, else at the closed end."""

    def __init__(self, inside="storage_drawer", heading=math.pi / 2, navigation=True, **results):
        self.robot = ROBOT
        self.position = (0.0, 0.0)
        self.calls, self.results, self.heading = [], {k: list(v) for k, v in results.items()}, heading
        self.steps = []  # The named steps, in the order they began, with their depth.
        self._depth = 0
        self.arm = np.zeros(len(JOINTS))  # Out of the stow.
        self.tool = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        target = TargetHint("target", (0.0, 2.4, 0.1), (0.04, 0.04, 0.12), inside, None if inside else "table")
        self.hints = SimpleNamespace(target=lambda: target, part=lambda name: DRAWER)
        self.navigation = SimpleNamespace(navigate_to=self._navigate) if navigation else None
        self.manipulation = SimpleNamespace(
            holding=False,
            episode_id=1,
            stow=self._stow,
            look_at=lambda target, views=None, held=None: self._arm_call("look_at", views),
            open_gripper=lambda: self._gripper(False),
            close_gripper=lambda: self._gripper(True),
            move_to_pose=self._move_to_pose,
            execute_trajectory=self._line,
            tool_pose=lambda tool="hand": tuple(self.tool),
            move_joints=lambda targets, duration: self._arm_call("move_joints"),
        )

    def _call(self, name, *details, kind=None):
        self.calls.append((name, *details))
        results = self.results.get(kind or name)
        return results.pop(0) if results else Feedback()

    def _arm_call(self, name, *details, kind=None):
        feedback = self._call(name, *details, kind=kind)
        if feedback != LINE_REFUSED and feedback.status == "succeeded":
            self.arm = np.zeros(len(JOINTS))
        return feedback

    def _stow(self, held=None):
        feedback = self._call("stow", held)
        if feedback.status == "succeeded":
            self.arm = np.asarray(ROBOT.stow)
        return feedback

    timeouts: list = []

    def _navigate(self, x, y, heading, timeout=None, **tolerance):
        self.timeouts = [*self.timeouts, timeout]
        self.heading, self.position = heading, (x, y)
        return self._call("navigate_to", (x, y, heading))

    def _gripper(self, close):
        self.manipulation.holding = close and self._near
        feedback = self._call("close_gripper" if close else "open_gripper")
        if feedback.gripper_aperture is None:
            stopped = (ON_BAR if self._near else ROBOT.gripper_closed) if close else ROBOT.gripper_open
            feedback = replace(feedback, gripper_aperture=stopped)
        return feedback

    _near = _fresh = False  # The jaws are at an object: after the first line from a planned move.

    def _move_to_pose(self, pose, *, held=None, tool="hand"):
        feedback = self._arm_call("approach", tuple(np.round(pose, 4)), held, tool)
        if feedback.status == "succeeded":
            self.tool = np.asarray(pose, dtype=float)
        self._near, self._fresh = False, feedback.status == "succeeded"
        return feedback

    def _line(self, line):
        start, end = np.asarray(line.poses[0]), np.asarray(line.poses[-1])
        motion = end[:3] - start[:3]
        distance = float(np.linalg.norm(motion))
        feedback = self._arm_call("line", tuple(np.round(motion / distance, 3)), round(distance, 3))
        if feedback == LINE_REFUSED:
            raise TrajectoryRefused("out_of_reach", "The trajectory leaves the arm's reach.")
        if feedback.status == "succeeded":
            self.tool, self._near, self._fresh = end, self._fresh, False
        return feedback

    def wait(self, feedback):
        return feedback

    def sleep(self, seconds):
        self.calls.append(("sleep", seconds))

    @contextmanager
    def step(self, name):
        self.steps.append((self._depth, name))
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def observe(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        x, y = self.position
        pose = SimpleNamespace(matrix=np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, BODY_Z], [0, 0, 0, 1.0]]))
        joints = SimpleNamespace(names=JOINTS, positions=self.arm.copy())
        return SimpleNamespace(localization=SimpleNamespace(valid=True, pose=pose), joints=joints)

    def names(self):
        return [call[0] for call in self.calls]

    def lines(self):
        return [call for call in self.calls if call[0] == "line"]


def test_the_retrieve_plan_opens_the_drawer_then_reaches_in_picks_and_stows():
    session = FakeSession()
    plan(session)
    releases = ["line"] * len(TUNING.handle_release)
    assert session.names() == [
        "stow", "navigate_to",  # The arm folds to walk to the open stance.
        "look_at",  # The handle.
        "open_gripper", "approach", "line", "close_gripper", "line", "open_gripper", *releases, "sleep",
        "stow", "navigate_to",  # The reach-in stance.
        "look_at", "open_gripper", "approach", "line", "close_gripper", "line", "sleep",  # Pick: in, lift.
        "stow",
    ]  # fmt: skip
    open_walk, reach_walk = (call[1] for call in session.calls if call[0] == "navigate_to")
    assert open_walk == pytest.approx(OPEN)
    assert reach_walk == pytest.approx((0.0, 2.4 - TUNING.reach_ahead_m, math.pi / 2))  # The target ahead.
    handle_view, target_view = (call[1] for call in session.calls if call[0] == "look_at")
    assert handle_view == TUNING.look_views["handle"] and target_view is None  # look_at's own views.
    into, slide, *release = session.lines()[: 2 + len(TUNING.handle_release)]
    assert into[2] == pytest.approx(TUNING.handle_approach_m)  # Straight onto the handle.
    assert slide[1] == (0.0, -1.0, 0.0) and slide[2] == pytest.approx(PULL_M)  # Until the grip range's near end.
    assert [call[2] for call in release] == [distance for _, distance in TUNING.handle_release]
    approach = np.asarray(into[1])  # Into the part.
    for (move, _), call in zip(TUNING.handle_release, release, strict=True):
        axis = np.asarray(call[1])
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-3)
        if move == "gripper_back":
            assert axis == pytest.approx(-approach, abs=1e-3)  # Straight back out along the gripper axis.
        if move == "wrist_down":
            assert abs(axis @ approach) < 1e-3
    pick_in, lift = session.lines()[-2:]
    distance = TUNING.approach_m
    assert pick_in[1:] == ((0.0, 0.0, -1.0), pytest.approx(distance))  # Straight down first.
    assert lift[1:] == ((0.0, 0.0, 1.0), pytest.approx(distance))
    held = session.calls[-1][1]  # The stow carries the picked object.
    assert held.name == "target" and held.size == (0.04, 0.04, 0.12)


def test_the_retrieve_plan_names_its_steps_for_the_timeline():
    session = FakeSession()
    plan(session)
    assert session.steps == [(0, "open_drawer"), (0, "pick(target)")]


def shoulder_pitch(handle, stance) -> float:
    """The pitch [deg] down along the line from Spot's shoulder to ``handle`` from the base at ``stance``."""
    x, y, heading = stance
    forward = np.array([math.cos(heading), math.sin(heading)])
    shoulder, height = np.array([x, y]) + forward * ROBOT.shoulder[0], BODY_Z + ROBOT.shoulder[2]
    return math.degrees(math.atan2(height - handle[2], float(np.linalg.norm(np.asarray(handle[:2]) - shoulder))))


def test_each_skill_reports_the_option_that_worked_and_what_it_measured():
    session = FakeSession()
    skills = CompositeSkills(session)
    opened = skills.open_drawer(DRAWER)
    pitch = shoulder_pitch(DRAWER.handle_center, OPEN)
    assert 0.0 < pitch < TUNING.handle_pitch_max_deg  # 18 deg: the grid's top drawer (18.5 deg in the sim).
    assert opened.option == dict(stance=0, stroke_m=pytest.approx(PULL_M), pitch_deg=pytest.approx(pitch, abs=0.05))
    assert opened.measured == dict(gripper=ON_BAR, moved_m=pytest.approx(PULL_M))
    assert opened.refused == 0
    picked = skills.pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12), name="target", part=DRAWER)
    stance = (0.0, 2.4 - TUNING.reach_ahead_m, math.pi / 2)
    assert picked.option == dict(
        approach=None,
        yaw_deg=0.0,
        stance=pytest.approx(stance, abs=1e-4),
        turn_deg=0.0,
        base=pytest.approx(stance, abs=1e-4),
    )
    assert picked.refused == 0 and picked.measured["holding"] and skills.held.name == "target"
    placed = skills.place((0.0, 2.0, 0.5))
    assert placed.option["turn_deg"] == 0.0 and len(placed.measured["released_at"]) == 3


def test_the_pick_uses_the_target_where_the_open_drawer_moved_it():
    session = FakeSession()
    hints = [TargetHint("target", (0.0, y, 0.1), (0.04, 0.04, 0.12), "storage_drawer", None) for y in (2.4, 2.1)]
    session.hints.target = lambda: hints.pop(0) if len(hints) > 1 else hints[0]  # Read again: opened.
    plan(session)
    pregrasp = [call[1] for call in session.calls if call[0] == "approach"][-1]
    assert abs(pregrasp[1] - 2.1) < 0.05  # Above the moved target, not where it was before.


def test_the_open_stance_is_the_same_rule_for_every_drawer():
    skills = CompositeSkills(FakeSession())
    assert skills.open_stance(DRAWER) == pytest.approx(OPEN)
    nearer = skills.open_stance(DRAWER, attempt=1)  # Standing again: one walk tolerance nearer.
    assert nearer == pytest.approx((0.0, OPEN[1] + TUNING.goal_tolerance[0], math.pi / 2))
    # A half-open drawer facing +x: the stance is measured from where its handle is now.
    turned = replace(DRAWER, handle_center=(3.2, 1.0, 0.3), direction=(1.0, 0.0, 0.0), opening=0.2)
    x, y, heading = skills.open_stance(turned)
    assert (x, y) == pytest.approx((3.2 + STANCE_M, 1.0))
    assert (math.cos(heading), math.sin(heading)) == pytest.approx((-1.0, 0.0))  # Facing -x.


def test_the_close_stance_centres_the_push_in_the_grip_range():
    """The hand starts push_back_m short of the open handle and moves the push's length: centred in
    the range, both ends keep the same margin."""
    session = FakeSession()
    opened = replace(DRAWER, handle_center=(0.0, 2.273, 0.465), opening=0.287)
    skills = CompositeSkills(session)
    skills.close_drawer(opened)
    stance = next(call[1] for call in session.calls if call[0] == "navigate_to")
    push = opened.opening - opened.limits[0] - TUNING.limit_margin_m
    start = 2.273 - stance[1] - TUNING.push_back_m  # The hand ahead of the base centre.
    near, far = TUNING.grip_range_m
    assert start - near == pytest.approx(far - (start + push))
    slide = session.lines()[1]
    assert slide[1] == (0.0, 1.0, 0.0) and slide[2] == pytest.approx(push)  # Into the carcass.


def test_a_drawer_place_spot_keeps_its_distance_from_the_carcass_whatever_the_opening():
    """The opened jaw reaches toward the carcass: at the floor's middle a 0.262 m opening left the
    box 0.127 m from it and the jaw caught its edge (2026-10-25). The spot moves out to the
    clearance, but no closer to the drawer front than the margin."""
    skills = CompositeSkills(FakeSession())
    size = (0.04, 0.04, 0.12)
    clearance, margin = TUNING.place_carcass_clearance_m, TUNING.place_front_margin_m
    for opening in (0.275, 0.284, 0.38):  # Wide enough: the clearance, or the middle if that is farther.
        floor = (0.0, 2.6 - opening / 2, 0.0125)  # The exposed floor's middle; the carcass front is at y 2.6.
        x, y, z = skills.place_spot(replace(DRAWER, opening=opening, floor_center=floor), size)
        assert (x, 2.6 - y, z) == pytest.approx((0.0, max(clearance, opening / 2), 0.0125))
    narrow = replace(DRAWER, opening=0.2, floor_center=(0.0, 2.5, 0.0125))  # Too narrow for the clearance.
    assert 2.6 - skills.place_spot(narrow, size)[1] == pytest.approx(0.2 - 0.02 - margin)


def test_a_target_on_a_table_is_reached_along_the_line_from_the_base():
    session = FakeSession(inside=None, heading=0.0)
    plan(session)
    stances = [call[1] for call in session.calls if call[0] == "navigate_to"]
    assert stances == [pytest.approx((0.0, 2.4 - TUNING.reach_ahead_m, math.pi / 2))]  # Facing the target.
    assert session.names().count("approach") == 1  # No drawer to open: only the pick's planned move.


def test_a_point_near_the_base_heading_is_reached_keeping_the_heading():
    """Square to the open drawer it stands at, the robot keeps its heading for a point within
    keep_heading_deg of it (16 deg off square, a house drawer pick closed on nothing twice)."""
    heading = math.pi / 2 + math.radians(20.0)
    session = FakeSession(inside=None, heading=heading)
    CompositeSkills(session).pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12))
    stance = next(call[1] for call in session.calls if call[0] == "navigate_to")
    ahead = TUNING.reach_ahead_m
    assert stance == pytest.approx((-ahead * math.cos(heading), 2.4 - ahead * math.sin(heading), heading))


def test_a_reach_stance_the_robot_does_not_fit_turns_about_the_point():
    """Facing a table's corner the body did not fit at the stance (drawer_place, 2026-11-01): the
    next stance turns about the point. Square to a drawer it does not turn."""
    unreachable = Feedback("unreachable", "Nav2 planner failed with code 206")
    session = FakeSession(inside=None, heading=0.0, navigate_to=[unreachable])
    picked = CompositeSkills(session).pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12))
    turned = math.pi / 2 + math.radians(TUNING.reach_turns_deg[1])
    stances = [call[1] for call in session.calls if call[0] == "navigate_to"]
    assert stances[1] == pytest.approx(
        (-TUNING.reach_ahead_m * math.cos(turned), 2.4 - TUNING.reach_ahead_m * math.sin(turned), turned)
    )
    assert picked.option["turn_deg"] == TUNING.reach_turns_deg[1]
    with pytest.raises(AgentFailed, match="unreachable"):
        CompositeSkills(FakeSession(navigate_to=[unreachable])).pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12), part=DRAWER)


def test_refused_pick_options_move_on_in_order():
    # The pick's straight-down approach is refused at every yaw, so it moves on to the first tilted
    # approach, given in the robot's frame.
    yaws = len(TUNING.grasp_yaws_deg)
    session = FakeSession(approach=[Feedback(), *[REFUSED] * yaws])  # The handle approach, then the pick's.
    plan(session)
    pick_in = session.lines()[-2]
    forward, side, down = TUNING.pick_approaches[1]  # In the robot's frame; the robot heads +y.
    assert pick_in[1] == pytest.approx((-side, forward, down), abs=1e-3)


def test_a_refused_handle_approach_stands_again_one_tolerance_nearer():
    """The one handle grasp is refused before moving: the skill stands again, nearer."""
    session = FakeSession(approach=[REFUSED])
    plan(session)
    walks = [call[1] for call in session.calls if call[0] == "navigate_to"]
    assert len(walks) == 3  # Open stance, again after the refusal, reach-in stance.
    assert walks[1] == pytest.approx((0.0, OPEN[1] + TUNING.goal_tolerance[0], math.pi / 2))
    assert session.names()[session.names().index("navigate_to", 2) - 1] == "stow"  # The arm folds to walk.
    with pytest.raises(AgentFailed, match=r"the handle approach was refused: .*, from 2 stances\."):
        plan(FakeSession(approach=[REFUSED] * TUNING.stance_tries))


def test_a_refused_pull_lets_go_backs_off_and_stands_again():
    """The gripper holds the handle and the pull leaves the arm's reach (Spot's upside-down hand,
    2026-11-03): it opens the gripper, backs off and stands again nearer."""
    session = FakeSession(line=[Feedback(), LINE_REFUSED])  # In, then the pull is refused.
    opened = CompositeSkills(session).open_drawer(DRAWER)
    names = session.names()
    second = names.index("navigate_to", names.index("navigate_to") + 1)
    releases = ["line"] * len(TUNING.handle_release)
    assert names[:second] == [
        "stow", "navigate_to", "look_at", "open_gripper", "approach", "line", "close_gripper",
        "line", "open_gripper", *releases, "stow",  # The refused pull; let go, back off, fold to walk.
    ]  # fmt: skip
    assert opened.option["stance"] == 1 and opened.refused == 1


@pytest.mark.parametrize(
    ("height", "expected"),
    [
        (0.155, TUNING.handle_pitch_max_deg),  # The grid's bottom drawer: 36.7 deg along the line, capped.
        (0.465, None),  # The grid's top drawer: along the line.
        (0.80, 0.0),  # Above the shoulder: level.
    ],
)
def test_the_handle_pitch_follows_the_line_from_the_shoulder_within_its_bounds(height, expected):
    session = FakeSession()
    drawer = replace(DRAWER, handle_center=(0.0, 2.56, height))
    opened = CompositeSkills(session).open_drawer(drawer)
    line = shoulder_pitch(drawer.handle_center, OPEN)
    assert opened.option["pitch_deg"] == pytest.approx(line if expected is None else expected, abs=0.05)
    into = np.asarray(session.lines()[0][1])  # The line onto the handle runs along the gripper axis.
    assert math.degrees(math.asin(-into[2])) == pytest.approx(opened.option["pitch_deg"], abs=0.1)


def test_a_gripper_stopped_short_of_the_bar_lets_go_and_fails():
    """Pitched 36.7 deg at a 0.155 m handle, Spot's jaw stopped on the drawer front at -1.33 rad
    (open -1.40); holding was true and the skill pulled the hand back alone (2026-11-03)."""
    session = FakeSession(close_gripper=[Feedback(gripper_aperture=-1.33)])
    with pytest.raises(AgentFailed, match="stopped 5% closed, not on the handle bar"):
        CompositeSkills(session).open_drawer(DRAWER)
    releases = ["line"] * len(TUNING.handle_release)
    assert session.names()[-2 - len(releases) :] == ["close_gripper", "open_gripper", *releases]  # No pull.


def test_holding_read_as_lost_after_the_pull_still_opens():
    """The jaws crept 0.039 rad of the 0.05 rad squeeze on the house island's bottom handle in the
    pull: holding read false with the bar held to the end and the drawer opened 0.255 m (retrieve_house,
    2026-11-03)."""
    session = FakeSession()
    line = session._line

    def creeping(trajectory):
        feedback = line(trajectory)
        if len(session.lines()) == 2:  # The pull.
            session.manipulation.holding = False
        return feedback

    session.manipulation.execute_trajectory = creeping
    opened = CompositeSkills(session).open_drawer(DRAWER)
    assert opened.measured["moved_m"] == pytest.approx(PULL_M)
    releases = ["line"] * len(TUNING.handle_release)
    assert session.names()[-2 - len(releases) :] == ["open_gripper", *releases, "sleep"]


def test_a_handle_approach_stopped_while_moving_backs_off_and_fails():
    """The approach's line stopped on contact (Spot's finger on the island's top drawer front, 154 N,
    2026-11-03): it backs off and reports the stop; no other grasp is tried."""
    session = FakeSession(line=[STALLED])
    with pytest.raises(AgentFailed, match="the handle approach stopped: End-effector tracking error"):
        CompositeSkills(session).open_drawer(DRAWER)
    releases = ["line"] * len(TUNING.handle_release)
    assert session.names()[-3 - len(releases) :] == ["approach", "line", "open_gripper", *releases]
    assert session.names().count("approach") == 1


def test_a_refused_release_move_is_tried_at_half_then_ends_the_release():
    moves = TUNING.handle_release
    assert len(moves) >= 2
    # In and the slide succeed; the first release move is refused at full and at half length.
    session = FakeSession(line=[Feedback(), Feedback(), LINE_REFUSED, LINE_REFUSED])
    plan(session)
    lines = session.lines()
    assert [call[2] for call in lines[2:4]] == pytest.approx([moves[0][1], moves[0][1] / 2])
    after = session.calls[session.calls.index(lines[3]) + 1]
    assert after[0] == "sleep"  # The later release moves were skipped.


def test_a_walk_navigation_does_not_finish_ends_the_skill():
    """navigate_to ends a walk itself (its fine approach, its corrections from the settled pose): no
    time limit and no second try on top."""
    session = FakeSession()
    plan(session)
    assert session.timeouts == [None, None]
    for status in ("timed_out", "blocked"):
        with pytest.raises(AgentFailed, match=status):
            plan(FakeSession(navigate_to=[Feedback(status, "Goal unreachable")]))


def test_a_drawer_stroke_goes_on_when_no_view_of_the_handle_is_reachable():
    """look_at tries the handle views itself; with none reachable the stroke goes on without the look.
    The arm failing while it moves ends the skill."""
    no_view = Feedback("failed", "No view of the target is reachable", SkillFailure.OUT_OF_REACH, progress=0.0)
    opened = CompositeSkills(FakeSession(look_at=[no_view])).open_drawer(DRAWER)
    assert opened.measured["moved_m"] == pytest.approx(PULL_M)
    with pytest.raises(AgentFailed, match="look_at"):
        CompositeSkills(FakeSession(look_at=[STALLED])).open_drawer(DRAWER)


def test_a_skill_stows_an_arm_that_is_out_before_it_walks_and_without_navigation_stays():
    session = FakeSession(inside=None)
    CompositeSkills(session).pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12))
    assert session.names()[:2] == ["stow", "navigate_to"]
    stowed = FakeSession(inside=None)
    stowed.arm = np.asarray(ROBOT.stow)
    CompositeSkills(stowed).pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12))
    assert stowed.names()[0] == "navigate_to"
    standing = FakeSession(inside=None, navigation=False)  # A tier without navigation: picks from here.
    CompositeSkills(standing).pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12))
    assert "navigate_to" not in standing.names() and "stow" not in standing.names()


def test_a_failure_while_moving_ends_the_plan():
    session = FakeSession(line=[Feedback(), STALLED])  # In, then the pull stalls.
    with pytest.raises(AgentFailed, match="the pull stopped after"):
        plan(session)
    assert session.names()[-1 - len(TUNING.handle_release)] == "open_gripper"  # It let go and backed off.
    before_lift = 2 + len(TUNING.handle_release) + 1  # Handle in, slide, releases, pick in.
    with pytest.raises(AgentFailed, match="pick lift"):
        plan(FakeSession(line=[Feedback()] * before_lift + [STALLED]))


def test_the_held_box_is_measured_in_the_wrist_after_the_jaws_close():
    """Closing on the box moved Spot's wrist 2.8 cm (pick_place_depth_moveit, 2026-11-01): measured
    before the close, the box sat that far off in the wrist frame and the place missed by 2.3 cm."""
    session = FakeSession(inside=None)
    close, closed_at = session.manipulation.close_gripper, []

    def closing():
        session.tool = session.tool + [0.02, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0]  # The jaw pushes the wrist.
        closed_at.append(session.tool.copy())
        return close()

    session.manipulation.close_gripper = closing
    skills = CompositeSkills(session)
    skills.pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12), name="target")
    session.tool = closed_at[0]
    expected = skills.held_object("target", np.array([0.0, 2.4, 0.1]), (0.04, 0.04, 0.12))
    assert skills.held.wrist_from_object == pytest.approx(expected.wrist_from_object)


def test_a_place_without_a_picked_object_or_that_keeps_holding_it_fails():
    session = FakeSession()
    skills = CompositeSkills(session)
    with pytest.raises(AgentFailed, match="no picked object"):
        skills.place((0.0, 2.0, 0.5))
    skills.pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12))
    placed = skills.place((0.0, 2.0, 0.5))
    assert skills.held is None and len(placed.measured["released_at"]) == 3


@pytest.mark.parametrize("robot", ["spot", "ridgeback_franka"])
def test_the_pick_grasp_pose_puts_the_grasp_frame_on_the_point_along_the_approach(robot):
    from phygs.robots.description import load_robot

    skills = CompositeSkills(SimpleNamespace(robot=load_robot(robot).info()))
    point = np.array([0.5, 0.2, 0.3])
    for approach in (DOWN, np.array([0.5, 0.0, -0.866])):
        approach = approach / np.linalg.norm(approach)
        wrist = skills.grasp_wrist_pose(point, approach, math.radians(90.0))
        assert skills.gripper_axes(wrist)[:, 0] == pytest.approx(approach, abs=1e-6)  # The gripper's approach.
        assert (wrist @ skills.tool("grasp"))[:3, 3] == pytest.approx(point, abs=1e-9)  # The grasp point reaches it.
    if robot == "spot":  # Spot's gripper axes are its wrist's: the wrist x is the approach.
        assert wrist[:3, 0] == pytest.approx(approach, abs=1e-6)
    else:  # The Franka hand approaches along its z.
        assert wrist[:3, 2] == pytest.approx(approach, abs=1e-6)


@pytest.mark.parametrize("side", [1.0, -1.0])
@pytest.mark.parametrize("pitch", [0.0, 15.0, 30.0])
def test_the_handle_grasp_holds_the_bar_along_the_jaws_closing_axis_up_pitched_down_into_the_drawer(side, pitch):
    skills = CompositeSkills(SimpleNamespace(robot=ROBOT))
    drawer = replace(DRAWER, direction=(0.6, -0.8, 0.0), handle_axis=(0.8 * side, 0.6 * side, 0.0))
    pose = skills.handle_grasp_pose(drawer, pitch)
    rotation = pose[:3, :3]
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[:, 1] == pytest.approx((-0.8, -0.6, 0.0), abs=1e-12)  # The bar along y, either sign given.
    assert rotation[2, 2] > 0  # The closing axis up: Spot's hand upright.
    into, down = -np.asarray(drawer.direction), math.radians(pitch)
    expected_axis = math.cos(down) * into + [0.0, 0.0, -math.sin(down)]  # Into the drawer, pitched down.
    assert rotation[:, 0] == pytest.approx(expected_axis, abs=1e-12)
    wrist = skills.wrist_for("handle", pose)
    bar = (np.linalg.inv(wrist) @ [*drawer.handle_center, 1.0])[:3]  # The bar's centre in the wrist frame.
    # Spot: in the jaws at x 0.22 m, resting on the fixed jaw (0.029 m below the wrist axis, plus the radius).
    assert bar == pytest.approx([0.22, 0.0, -0.029 + drawer.handle_radius], abs=1e-12)
    assert abs(np.dot(quaternion(rotation), Rotation.from_matrix(rotation).as_quat())) == pytest.approx(1.0)


def test_reaching_into_an_open_drawer_stops_clear_of_it_first():
    """The reach stance puts the point reach_ahead_m ahead; when that puts the body over the open
    drawer (its handle nearer than reach_in_clear_m), a stance with the handle that far ahead comes
    first, on the same line (navigation's fine approach makes the last bit)."""
    skills = CompositeSkills(FakeSession())
    opened = replace(DRAWER, handle_center=(0.0, 2.21, 0.465), opening=0.35)
    clear, stance = skills.reach_stances((0.0, 2.5, 0.44), part=opened)
    assert stance == pytest.approx((0.0, 2.5 - TUNING.reach_ahead_m, math.pi / 2))
    assert clear == pytest.approx((0.0, 2.21 - TUNING.reach_in_clear_m, math.pi / 2))
    assert len(skills.reach_stances((0.0, 2.3, 0.44), part=opened)) == 1  # Clear of it already.
