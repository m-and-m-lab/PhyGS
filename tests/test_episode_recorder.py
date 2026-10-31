# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode contact records: the grasp apart from other arm contacts."""

from phygs.world.recorder import Recorder, pinched_bodies

JAWS = ("arm_link_wr1", "arm_link_fngr")


def recorder(tmp_path):
    return Recorder(tmp_path, video=False, fps=10.0, jaw_links=JAWS)


def summary(records):
    return sorted((r["link"], r["body"], r["peak_n"], r["ticks"]) for r in records)


def test_pinched_bodies_are_the_ones_every_jaw_presses_on():
    contacts = [("arm_link_wr1", "box", 158.5), ("arm_link_fngr", "box", 145.5), ("arm_link_wr1", "drawer", 12.0)]
    assert pinched_bodies(contacts, JAWS) == {"box"}
    assert pinched_bodies(contacts[:1], JAWS) == set()


def test_a_hold_counts_every_jaw_contact_with_the_pinched_body_as_the_grasp(tmp_path):
    """The finger lands before the fixed jaw, then a pull loads one jaw only (drawer strokes, 2026-10-23)."""
    rec = recorder(tmp_path)
    rec.contacts_tick(1.0, [("arm_link_fngr", "drawer", 134.7)], "close_gripper", gripper_closed=True)
    rec.contacts_tick(1.1, [("arm_link_fngr", "drawer", 150.0), ("arm_link_wr1", "drawer", 210.9)], "close_gripper",
                      gripper_closed=True)  # fmt: skip
    rec.contacts_tick(2.0, [("arm_link_wr1", "drawer", 79.4)], "execute_trajectory", gripper_closed=True)
    rec.contacts_tick(2.1, [("arm_link_el1", "drawer", 3.0)], "execute_trajectory", gripper_closed=True)  # Not a jaw.
    rec.contacts_tick(3.0, [("arm_link_wr1", "drawer", 30.9)], "open_gripper", gripper_closed=False)  # After release.
    report = rec.contact_report()
    assert summary(report["grasp_contacts"]) == [
        ("arm_link_fngr", "drawer", 150.0, 2),
        ("arm_link_wr1", "drawer", 210.9, 2),
    ]
    assert summary(report["contacts"]) == [("arm_link_el1", "drawer", 3.0, 1), ("arm_link_wr1", "drawer", 30.9, 1)]


def test_without_a_pinch_or_a_closed_gripper_nothing_is_the_grasp(tmp_path):
    rec = recorder(tmp_path)
    # A closed hand bumping a table with one jaw, then an open hand pressing both jaws on it.
    rec.contacts_tick(1.0, [("arm_link_wr1", "table", 20.0)], "move_to_pose", gripper_closed=True)
    rec.contacts_tick(2.0, [("arm_link_wr1", "table", 40.0), ("arm_link_fngr", "table", 35.0)], "move_to_pose",
                      gripper_closed=False)  # fmt: skip
    report = rec.contact_report()
    assert report["grasp_contacts"] == []
    assert summary(report["contacts"]) == [("arm_link_fngr", "table", 35.0, 1), ("arm_link_wr1", "table", 40.0, 2)]


def test_each_hold_is_judged_on_its_own(tmp_path):
    rec = recorder(tmp_path)
    pinch = [("arm_link_wr1", "box", 120.0), ("arm_link_fngr", "box", 110.0)]
    rec.contacts_tick(1.0, pinch, "close_gripper", gripper_closed=True)
    rec.contacts_tick(2.0, [], "open_gripper", gripper_closed=False)
    rec.contacts_tick(3.0, [("arm_link_wr1", "box", 15.0)], "move_to_pose", gripper_closed=True)  # Closed, no pinch.
    report = rec.contact_report()
    assert summary(report["grasp_contacts"]) == [("arm_link_fngr", "box", 110.0, 1), ("arm_link_wr1", "box", 120.0, 1)]
    assert summary(report["contacts"]) == [("arm_link_wr1", "box", 15.0, 1)]


def test_a_read_counts_as_the_ticks_between_reads(tmp_path):
    """The World reads contacts every few physics ticks; ``ticks`` stays in physics ticks."""
    rec = Recorder(tmp_path, video=False, fps=10.0, jaw_links=JAWS, read_ticks=5)
    for time_seconds in (1.0, 1.02):
        rec.contacts_tick(time_seconds, [("arm_link_el1", "table", 2.0)], "move_to_pose", gripper_closed=False)
    assert summary(rec.contact_report()["contacts"]) == [("arm_link_el1", "table", 2.0, 10)]


def test_contacts_so_far_sort_the_current_hold_without_ending_it(tmp_path):
    rec = recorder(tmp_path)
    rec.contacts_tick(1.0, [("arm_link_fngr", "box", 30.0)], "close_gripper", gripper_closed=True)
    rec.contacts_tick(1.1, [("arm_link_wr1", "box", 40.0), ("arm_link_fngr", "box", 45.0)], "lift", gripper_closed=True)
    so_far = rec.contacts_so_far()
    assert summary(so_far["grasp_contacts"]) == [("arm_link_fngr", "box", 45.0, 2), ("arm_link_wr1", "box", 40.0, 1)]
    rec.contacts_tick(1.2, [("arm_link_fngr", "box", 50.0)], "lift", gripper_closed=True)  # The hold goes on.
    report = rec.contact_report()
    assert summary(report["grasp_contacts"]) == [("arm_link_fngr", "box", 50.0, 3), ("arm_link_wr1", "box", 40.0, 1)]
    assert so_far["grasp_contacts"][0]["peak_n"] in (45.0, 40.0)  # The earlier copy is unchanged.
