# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The composite skills and the robot tests' retrieve plan on a fake session: order, retries, the
options and measurements they report, and geometry (no Isaac Sim)."""

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
ROBOT = SPOT.info()


class FakeSession:
    """Clients return feedback as their command id; ``wait`` hands it back. Scripted results are
    consumed in order per kind (``approach``: planned moves, ``line``: end-effector lines,
    ``navigate_to``, ``stow``); the rest succeed. The tool pose follows the moves that succeed; the
    arm joints are at the stow posture after a stow and out after any other arm move."""

    def __init__(self, inside="storage_drawer", heading=math.pi / 2, **results):
        self.robot = ROBOT
        self.position = (0.0, 0.0)
        self.calls, self.results, self.heading = [], {k: list(v) for k, v in results.items()}, heading
        self.steps = []  # The named steps, in the order they began, with their depth.
        self._depth = 0
        self.arm = np.zeros(len(JOINTS))  # Out of the stow.
        self.tool = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        target = TargetHint("target", (0.0, 2.4, 0.1), (0.04, 0.04, 0.12), inside, None if inside else "table")
        self.hints = SimpleNamespace(target=lambda: target, part=lambda name: DRAWER)
        self.navigation = SimpleNamespace(navigate_to=self._navigate)
        self.manipulation = SimpleNamespace(
            holding=False,
            episode_id=1,
            stow=self._stow,
            look_at=lambda target, **view: self._arm_call("look_at"),
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
        return self._call("close_gripper" if close else "open_gripper")

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
        pose = SimpleNamespace(matrix=np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
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
    assert reach_walk == pytest.approx((0.0, 2.4 - TUNING.reach_ahead_m, math.pi / 2))  # The target ahead.
    into, slide, *release = session.lines()[: 2 + len(TUNING.handle_release)]
    assert into[2] == pytest.approx(TUNING.handle_approach_m[0])  # Straight onto the handle.
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
    assert session.steps == [
        (0, "stow"),
        (0, "open_drawer"),
        (1, "walk"),
        (1, "look_at(handle)"),
        (0, "stow"),
        (0, "go_to_reach"),
        (1, "walk"),
        (0, "pick(target)"),
        (1, "look_at(target)"),  # Inside the pick.
        (0, "stow"),
    ]


def test_each_skill_reports_the_option_that_worked_and_what_it_measured():
    session = FakeSession(approach=[REFUSED, Feedback()])
    skills = CompositeSkills(session)
    opened = skills.open_drawer(DRAWER)
    assert opened.option == dict(stance=0, stroke_m=pytest.approx(PULL_M), pitch_deg=30.0, approach_m=0.08, side=-1.0)
    assert opened.measured == dict(moved_m=pytest.approx(PULL_M), held_handle=True)
    skills.stow()
    skills.go_to_reach((0.0, 2.4, 0.1), part=DRAWER)
    picked = skills.pick((0.0, 2.4, 0.1), (0.04, 0.04, 0.12), name="target")
    assert picked.option == dict(approach=None, yaw_deg=0.0) and picked.refused == 0
    assert picked.measured["holding"] and skills.held.name == "target"
    walked = skills.walk((1.0, 2.0, 0.0))
    assert walked.option == dict(stance=[1.0, 2.0, 0.0], attempt=1)
    assert walked.measured["base"] == pytest.approx([1.0, 2.0, 0.0])


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


def test_a_reach_stance_the_robot_does_not_fit_turns_about_the_point():
    """Facing a table's corner the body did not fit at the stance (drawer_place, 2026-11-01): the
    next stance turns about the point. Square to a drawer it does not turn."""
    unreachable = Feedback("unreachable", "Nav2 planner failed with code 206")
    session = FakeSession(inside=None, heading=0.0, navigate_to=[unreachable])
    reached = CompositeSkills(session).go_to_reach((0.0, 2.4, 0.1))
    turned = math.pi / 2 + math.radians(TUNING.reach_turns_deg[1])
    stances = [call[1] for call in session.calls if call[0] == "navigate_to"]
    assert stances[1] == pytest.approx(
        (-TUNING.reach_ahead_m * math.cos(turned), 2.4 - TUNING.reach_ahead_m * math.sin(turned), turned)
    )
    assert reached.refused == 1 and reached.option["turn_deg"] == TUNING.reach_turns_deg[1]
    with pytest.raises(AgentFailed, match="unreachable"):
        CompositeSkills(FakeSession(navigate_to=[unreachable])).go_to_reach((0.0, 2.4, 0.1), part=DRAWER)


def test_refused_options_move_on_in_order():
    # The first handle approach (both bar directions at the first distance) is refused; the pick's
    # straight-down approach is refused at every yaw, so it moves on to the first tilted approach,
    # given in the robot's frame.
    yaws = len(TUNING.grasp_yaws_deg)
    session = FakeSession(approach=[REFUSED, REFUSED, Feedback(), *[REFUSED] * yaws])
    plan(session)
    handle_in = session.lines()[0]
    assert handle_in[2] == pytest.approx(TUNING.handle_approach_m[1])  # 0.06 m: the next distance.
    pick_in = session.lines()[-2]
    forward, side, down = TUNING.pick_approaches[1]  # In the robot's frame; the robot heads +y.
    assert pick_in[1] == pytest.approx((-side, forward, down), abs=1e-3)


def test_when_every_pitch_is_refused_the_drawer_skill_stands_again_one_tolerance_nearer():
    refusals = [REFUSED] * (2 * len(TUNING.handle_approach_m) * len(TUNING.handle_pitch_deg))
    session = FakeSession(approach=refusals)
    plan(session)
    walks = [call[1] for call in session.calls if call[0] == "navigate_to"]
    assert len(walks) == 3  # Open stance, again after the refusals, reach-in stance.
    assert walks[1] == pytest.approx((0.0, OPEN[1] + TUNING.goal_tolerance[0], math.pi / 2))
    assert session.names()[session.names().index("navigate_to", 2) - 1] == "stow"  # The arm folds to walk.
    with pytest.raises(AgentFailed, match="every handle pitch"):
        plan(FakeSession(approach=refusals * TUNING.stance_tries))


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


def test_a_timed_out_walk_is_issued_again_once():
    timed_out = Feedback("timed_out", "Navigation time budget exhausted")
    session = FakeSession(navigate_to=[timed_out])
    plan(session)
    stances = [call for call in session.calls if call[0] == "navigate_to"]
    assert stances[0][1] == stances[1][1] == pytest.approx(OPEN)
    assert session.timeouts[0] == TUNING.walk_timeout_s
    with pytest.raises(AgentFailed, match="timed_out"):
        plan(FakeSession(navigate_to=[timed_out] * TUNING.walk_tries))
    with pytest.raises(AgentFailed, match="blocked"):
        plan(FakeSession(navigate_to=[Feedback("blocked", "Goal unreachable")]))


def test_a_look_at_tries_each_view_and_reports_when_none_was_reachable():
    out_of_reach = Feedback("failed", "out of reach", SkillFailure.OUT_OF_REACH, progress=0.0)
    views = TUNING.look_views["handle"]
    session = FakeSession(look_at=[out_of_reach])
    looked = CompositeSkills(session).look_at((0.0, 2.0, 0.5), "handle")
    assert looked.succeeded and looked.refused == 1
    assert looked.option == dict(distance_m=views[1][0], elevation_deg=views[1][1])
    session = FakeSession(look_at=[out_of_reach] * len(views))
    looked = CompositeSkills(session).look_at((0.0, 2.0, 0.5), "handle")
    assert not looked.succeeded and "no view" in looked.reason and session.names().count("look_at") == len(views)
    with pytest.raises(AgentFailed, match="look_at"):
        CompositeSkills(FakeSession(look_at=[STALLED])).look_at((0.0, 2.0, 0.5))  # Failed while moving.


def test_a_walk_stows_an_arm_that_is_out_first():
    session = FakeSession()
    CompositeSkills(session).walk((1.0, 0.0, 0.0))
    assert session.names() == ["stow", "navigate_to", "sleep"]


def test_a_stow_without_a_plan_from_where_the_arm_is_lifts_the_hand_and_stows_again():
    no_plan = Feedback("failed", "Pose planner failed: no_plan_within_budget", SkillFailure.PLANNING_FAILED, progress=0)
    session = FakeSession(stow=[no_plan])
    stowed = CompositeSkills(session).stow()
    assert session.names() == ["stow", "line", "stow"] and stowed.option == dict(how="lift, then stow")
    assert session.lines()[0][1:] == ((0.0, 0.0, 1.0), pytest.approx(TUNING.unstick_m))
    with pytest.raises(AgentFailed, match="stow"):
        CompositeSkills(FakeSession(stow=[no_plan, no_plan])).stow()


def test_a_failure_while_moving_ends_the_plan():
    with pytest.raises(AgentFailed, match="slide"):
        plan(FakeSession(line=[Feedback(), STALLED]))  # In, then the slide stalls.
    before_lift = 2 + len(TUNING.handle_release) + 1  # Handle in, slide, releases, pick in.
    with pytest.raises(AgentFailed, match="pick lift"):
        plan(FakeSession(line=[Feedback()] * before_lift + [STALLED]))


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
@pytest.mark.parametrize("pitch", [0.0, 15.0, 30.0, 45.0])
def test_the_handle_grasp_holds_the_bar_along_the_jaws_pitched_down_into_the_drawer(side, pitch):
    skills = CompositeSkills(SimpleNamespace(robot=ROBOT))
    drawer = replace(DRAWER, direction=(0.6, -0.8, 0.0), handle_axis=(0.8, 0.6, 0.0))
    pose = skills.handle_grasp_pose(drawer, side, pitch)
    rotation = pose[:3, :3]
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[:, 1] == pytest.approx(np.multiply(drawer.handle_axis, side), abs=1e-12)  # The bar along y.
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
