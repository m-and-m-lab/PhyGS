# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The episode recording (phygs.recording.log): what it writes, and that it never fails a run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

rr = pytest.importorskip("rerun")

from phygs.recording import log as episode_log  # noqa: E402
from phygs.recording.log import EpisodeLog  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
SPOT_URDF = PROJECT / "phygs" / "robots" / "spot" / "assets" / "spot.urdf"


def entities(path: Path) -> dict[str, int]:
    """Entity path -> rows written, read back from the file."""
    rows: dict[str, int] = {}
    for chunk in rr.chunk.RrdReader(path).stream().to_chunks():
        rows[str(chunk.entity_path)] = rows.get(str(chunk.entity_path), 0) + chunk.num_rows
    return rows


def test_a_recording_holds_the_robot_the_scene_the_calls_and_the_phases(tmp_path):
    path = tmp_path / episode_log.FILE
    log = EpisodeLog(path, run_id="run-1")
    joints = ("arm_sh0", "front_left_knee", "not_in_the_urdf")
    log.robot_model(SPOT_URDF, joints)
    log.scene([(None, np.zeros(3), np.ones(3)), ("/World/box", np.zeros(3), np.full(3, 0.2))], ["/World/box"])
    for tick in range(400):  # 1.6 s of physics ticks: one batch sent on the way, one at close.
        log.at(tick, tick * 0.004)
        positions = torch.tensor([0.1, -1.2, 0.001 * tick])
        log.joint_state(positions, positions + 0.01, torch.full((3,), 2.0))
        log.joint_state(positions, positions, positions)  # Again in the same tick: ignored.
        if tick % 8 == 0:  # A state sample every camera period.
            log.robot_state(np.array([tick * 0.001, 0, 0.5, 0, 0, 0, 1]), positions.numpy())
            log.body_poses(np.array([[1.0 + (0.01 if tick >= 200 else 0.0), 0, 0.4, 0, 0, 0, 1]]))
    log.agent_call(dict(t=0.4, call="navigation.navigate_to", args={"x": 1.0}, result="accepted"))
    log.phase("agent")
    for forces in ([("arm_link_fngr", "/World/box", 12.5)], [("arm_link_fngr", "/World/box", 14.0)], [], []):
        log.contacts(forces, "pick")  # Touches, pushes harder, lets go; then nothing to log.
    log.command_span(0, dict(lane="base", name="navigate_to", start=0.1, end=0.9, status="succeeded"))
    report = log.close()
    assert report == dict(status="complete", file=episode_log.FILE, error=None, schema=1, run_id="run-1")
    rows = entities(path)
    for entity in ("/truth/spot/joint_positions", "/truth/spot/joint_efforts", "/control/joint_targets"):
        assert rows[entity] == 400 + 1, entity  # Every physics tick (and the lines' names, once).
    assert rows["/truth/spot/joints/arm_sh0"] == 50 and "/truth/spot/joints/not_in_the_urdf" not in rows
    assert not [name for name in rows if "visual_geometries" in name]  # The collision geometry only.
    assert any("collision_geometries" in name for name in rows)
    # Only robot_state moves the joints: nothing fixes them at zero (Rerun's URDF logging would).
    assert "/tf_static" not in rows and rows["/truth/spot/joints/front_rail_joint"] == 1  # A fixed joint, once.
    assert rows["/truth/spot/base"] == 50
    body = next(name for name in rows if name.startswith("/truth/scene/bodies/") and not name.endswith("colliders"))
    assert rows[body] == 2  # The start, then the 1 cm move.
    assert rows["/agent/calls"] == 1 and rows["/events/phase"] == 1 and rows["/truth/contact_events"] == 1
    assert rows["/truth/contact_forces/arm_link_fngr/\\/World\\/box"] == 3  # 12.5 N, 14 N, then 0 N.
    assert rows["/control/commands"] == 1


def test_the_joint_transforms_equal_rerun_s_own_urdf_computation():
    tree = rr.urdf.UrdfTree.from_file_path(str(SPOT_URDF), "truth")
    values = np.array([-0.7, 0.0, 0.4])
    for joint in tree.joints():
        if joint.joint_type == "fixed":
            continue
        translation, quaternion = episode_log._Joint.of(joint).transforms(values)
        for row, value in enumerate(values):
            expected = joint.compute_transform(float(value), clamp=False)
            want_q = np.asarray(expected.quaternion.as_arrow_array().to_pylist()[0])
            want_t = np.asarray(expected.translation.as_arrow_array().to_pylist()[0])
            same = np.allclose(quaternion[row], want_q, atol=1e-6) or np.allclose(quaternion[row], -want_q, atol=1e-6)
            assert same, joint.name
            assert np.allclose(translation[row], want_t, atol=1e-6), joint.name


def test_a_logging_error_closes_the_recording_and_is_reported_but_never_raised(tmp_path):
    log = EpisodeLog(tmp_path / episode_log.FILE, run_id="run-2")
    log.robot_model(tmp_path / "missing.urdf", ())  # Fails inside: no such file.
    log.phase("agent")  # Ignored after the failure.
    report = log.close()
    assert report["status"] == "failed" and report["error"].startswith("robot_model:")


class _Misused(EpisodeLog):
    @episode_log._guarded
    def misuse(self) -> None:
        self._stream.send_columns(["a", "list"], indexes=[], columns=[])  # Rerun only warns, unless strict.


def test_a_call_rerun_refuses_fails_the_recording_instead_of_dropping_the_rows(tmp_path):
    log = _Misused(tmp_path / episode_log.FILE, run_id="run-5")
    log.misuse()
    report = log.close()
    assert report["status"] == "failed" and report["error"].startswith("misuse:")


def test_report_only_writes_no_recording(tmp_path):
    log = EpisodeLog(tmp_path / episode_log.FILE, run_id="run-3", level="report_only")
    log.phase("agent")
    assert log.close()["status"] == "off" and not (tmp_path / episode_log.FILE).exists()


def test_the_properties_carry_the_schema_and_the_run_id(tmp_path):
    path = tmp_path / episode_log.FILE
    log = EpisodeLog(path, run_id="run-4")
    log.properties(spec={"id": "dev_grid_table"})
    log.close()
    rows = entities(path)
    properties = [name for name in rows if name.startswith(rr.RECORDING_PROPERTIES_PATH)]
    assert any(name.endswith("phygs_log_schema") for name in properties), properties
    assert any(name.endswith("spec") for name in properties), properties
