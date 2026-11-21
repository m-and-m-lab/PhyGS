# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The task evaluator's record in the episode's recording (``/eval``), through the World's
:class:`~phygs.world.task.Evaluation` with a fake task: an entry after the reset and every period, the
result at the end, each with the run id; none of it, and nothing of ``/truth``, in a public copy."""

from __future__ import annotations

import json

import pytest

rr = pytest.importorskip("rerun")

from phygs.recording import log as episode_log  # noqa: E402
from phygs.recording.log import EpisodeLog  # noqa: E402
from phygs.recording.reader import Recording, public_copy  # noqa: E402
from phygs.world.task import EpisodeEnd, Evaluation, NoTask  # noqa: E402

PERIOD = 0.2  # [s]; the evaluation period of the fake run.


class FakeTask(NoTask):
    """Walks 0.5 m per period; reaches the target at the third period and lifts it at the fifth."""

    goal = "retrieve"

    def reset(self, truth):
        self.samples, self.path = 0, 0.0

    def update(self, truth):
        self.samples += 1
        self.path += 0.5

    def trace(self):
        return dict(path_length_m=self.path, progress=dict(reached=self.samples >= 3, lifted=self.samples >= 5))

    def result(self, truth, end):
        return dict(outcome="success", measured=dict(path_length_m=self.path))


def record(path, task, periods: int = 6) -> tuple[dict, list[dict]]:
    """One fake episode recorded at ``path``: reset, ``periods`` updates, the result; returns the
    report's recording entry and the evaluation."""
    log = EpisodeLog(path, run_id="run-7")
    log.at(0, 0.0)
    log._stamp()
    log._stream.log("/truth/scene/bodies/\\/World\\/box", rr.Scalars(1.0))  # An escaped entity path too.
    evaluation = Evaluation(task, log)
    evaluation.reset(truth=None)
    for k in range(1, periods + 1):
        log.at(k * 50, k * PERIOD)
        evaluation.update(truth=None)
    end = EpisodeEnd("finish", periods * PERIOD, ())
    result = evaluation.result(None, end)
    evaluation.end(result, end)
    log.phase("evaluation")
    return log.close(), result


def records(recording: Recording, *entity: str) -> list[tuple[float, dict]]:
    """(sim time, the JSON record) of each row of an ``/eval`` entity."""
    return [(seconds, json.loads(values["record"][0])) for _, seconds, values in recording.rows(*entity)]


def test_the_evaluator_s_state_is_recorded_after_the_reset_every_period_and_at_the_end(tmp_path):
    path = tmp_path / episode_log.FILE
    report, result = record(path, FakeTask())
    assert report["status"] == "complete"
    recording = Recording(path)
    states = records(recording, "eval", "state")
    # After the reset, each period, and once more with the result (the state it was scored from).
    assert [round(t, 3) for t, _ in states] == [round(k * PERIOD, 3) for k in range(7)] + [1.2]
    assert all(item["run_id"] == "run-7" for _, item in states)
    assert [item["state"]["path_length_m"] for _, item in states] == [0.5 * k for k in range(7)] + [3.0]
    lengths = [values for _, _, values in recording.rows("eval", "series", "path_length_m")]
    assert len(lengths) == 8
    reached = recording.rows("eval", "series", "progress", "reached")
    assert [values["Scalars:scalars"][0] for _, _, values in reached] == [0, 0, 0, 1, 1, 1, 1, 1]
    stages = [(round(t, 3), values) for _, t, values in recording.rows("eval", "stages")]
    assert [t for t, _ in stages] == [0.6, 1.0]  # Each stage once, when first reached.
    ((t, end),) = records(recording, "eval", "result")
    assert end["run_id"] == "run-7" and end["result"] == result
    assert end["end"] == dict(ended_by="finish", sim_seconds=pytest.approx(1.2))
    assert end["first_true"] == {"progress/reached": pytest.approx(0.6), "progress/lifted": pytest.approx(1.0)}


def test_only_finite_numbers_and_flags_become_series(tmp_path):
    """Text, None, NaN and a number too large for a float stay in the JSON state, never in a series,
    and never close the recording."""
    path = tmp_path / episode_log.FILE

    class Odd(FakeTask):
        def trace(self):
            return dict(path_length_m=self.path, note="text", missing=None, ratio=float("nan"), huge=10**400)

    report, _ = record(path, Odd(), periods=1)
    recording = Recording(path)
    assert report["status"] == "complete"
    assert {entity[2] for entity in recording.entities("eval", "series")} == {"path_length_m"}
    assert recording.rows("eval", "state")[0][2]["record"][0].count('"note": "text"') == 1


def test_a_task_without_a_trace_still_has_its_result_recorded(tmp_path):
    path = tmp_path / episode_log.FILE

    class Plain(FakeTask):
        trace = None  # No running state to record.

    record(path, Plain())
    recording = Recording(path)
    assert recording.rows("eval", "state") == [] and len(recording.rows("eval", "result")) == 1


def test_a_trace_that_raises_is_recorded_once_and_never_fails_the_episode(tmp_path):
    path = tmp_path / episode_log.FILE

    class Broken(FakeTask):
        def trace(self):
            raise KeyError("progress")

    report, result = record(path, Broken())
    assert report["status"] == "complete" and result["outcome"] == "success"
    states = records(Recording(path), "eval", "state")
    assert [item["state"] for _, item in states] == [{"trace_error": "KeyError: 'progress'"}]


def test_a_public_copy_has_neither_the_truth_nor_the_evaluation(tmp_path):
    path = tmp_path / episode_log.FILE
    record(path, FakeTask())
    branches = {entity[0] for entity in Recording(path).entities()}
    assert {"truth", "eval", "events"} <= branches
    copy = public_copy(tmp_path, tmp_path / "public.rrd")
    left = Recording(copy)
    assert {entity[0] for entity in left.entities()} == branches - {"truth", "eval"}
    assert left.properties["run_id"] == "run-7"
