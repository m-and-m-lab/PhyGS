# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import pytest

from phygs.scene import registry


def fetched(root, scene_id, version, commit):
    """Install a fake release as the fetcher does: its file, its record, and the id's current record."""
    usd = root / commit / scene_id / "scene.usdc"
    usd.parent.mkdir(parents=True)
    usd.write_text("")
    record = dict(scene_id=scene_id, scene_version=version, usd_path=str(usd))
    (root / "records" / commit).mkdir(parents=True)
    (root / "records" / commit / f"{scene_id}-{version}.json").write_text(json.dumps(record))
    (root / f"{scene_id}.json").write_text(json.dumps(record))
    return usd


def test_a_spec_pinned_to_an_earlier_fetched_release_still_finds_it(tmp_path, monkeypatch):
    monkeypatch.setenv(registry.SCENES_ENV, str(tmp_path))
    old = fetched(tmp_path, "house", "1.1.0", "c1")
    new = fetched(tmp_path, "house", "1.2.0", "c2")  # The last fetch: the id's current record.
    assert registry.scene_file("house", "1.1.0") == old.resolve()
    assert registry.scene_file("house", "1.2.0") == new.resolve()
    assert registry.scene_file("house") == new.resolve()
    with pytest.raises(ValueError, match="the spec wants 1.0.0: fetch it"):
        registry.scene_file("house", "1.0.0")
