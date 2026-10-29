# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode provenance: asset content hashes and the code version."""

import subprocess

from phygs.world.provenance import code_version, layers_sha256, usd_sha256


def test_an_edit_to_a_composed_layer_changes_the_robot_hash(tmp_path):
    (tmp_path / "physics.usda").write_text('#usda 1.0\ndef "Robot"\n{\n    double mass = 32.7\n}\n')
    (tmp_path / "robot.usda").write_text("#usda 1.0\n(\n    subLayers = [@./physics.usda@]\n)\n")
    first = usd_sha256(tmp_path / "robot.usda")
    assert first["layers"] == 2 and first["remote"] == 0
    assert usd_sha256(tmp_path / "robot.usda") == first
    (tmp_path / "physics.usda").write_text('#usda 1.0\ndef "Robot"\n{\n    double mass = 16.4\n}\n')  # Halved.
    assert usd_sha256(tmp_path / "robot.usda")["sha256"] != first["sha256"]


def test_a_layer_on_an_asset_server_counts_by_its_url(tmp_path):
    """The robot's camera models are layers on the Isaac asset server (an https URL, 2026-10-23)."""
    (tmp_path / "robot.usda").write_text("#usda 1.0\n")
    url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/camera.usd"
    local = layers_sha256([str(tmp_path / "robot.usda")])
    both = layers_sha256([str(tmp_path / "robot.usda"), url])
    assert both["layers"] == 2 and both["remote"] == 1 and both["sha256"] != local["sha256"]
    assert layers_sha256([url, str(tmp_path / "robot.usda")]) == both  # Order does not matter.


def test_code_version_reports_the_commit_and_uncommitted_edits(tmp_path):
    assert code_version(tmp_path) == dict(commit=None, dirty=None)  # Not a checkout.

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    clean = code_version(tmp_path)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False
    (tmp_path / "notes.txt").write_text("untracked\n")  # Untracked files are not code changes.
    assert code_version(tmp_path)["dirty"] is False
    (tmp_path / "a.py").write_text("x = 2\n")
    assert code_version(tmp_path) == dict(commit=clean["commit"], dirty=True)


def test_the_report_schema_lists_every_key_the_world_writes():
    """The keys in ``report.py`` and the keys ``World.run`` composes are the same set, and
    :func:`compose` refuses a report with one missing or one extra."""
    import ast
    from pathlib import Path

    import pytest

    from phygs.world.report import KEYS, REPORT_SCHEMA, compose

    tree = ast.parse((Path(__file__).resolve().parents[1] / "phygs" / "world" / "world.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compose"
    ]
    assert len(calls) == 1
    written = {keyword.arg for keyword in calls[0].keywords if keyword.arg}
    written |= {"contacts", "grasp_contacts"}  # From **contacts.
    assert REPORT_SCHEMA == 2 and set(KEYS) == written, sorted(set(KEYS) ^ written)
    fields = dict.fromkeys(KEYS)
    assert list(compose(**fields)) == list(KEYS)
    with pytest.raises(ValueError, match="missing"):
        compose(**{key: None for key in KEYS[1:]})
    with pytest.raises(ValueError, match="not in KEYS"):
        compose(**fields, extra=1)


def test_the_phygs_command_dispatches_and_prints_its_version(capsys):
    from phygs import __version__
    from phygs.__main__ import SCRIPTS, main

    assert main(["version"]) == 0 and capsys.readouterr().out.strip() == __version__
    assert main([]) == 2 and main(["nothing"]) == 2
    assert all(path.is_file() for path in SCRIPTS.values())
