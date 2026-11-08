# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan a finished run's arm plans again, offline, from its recording: every request the arm planner
sent, with the depth images it saw and the planner model it used, through a planner worker of this
checkout (no Isaac Sim). Use it to compare planner versions or settings on real requests.

    CUDA_VISIBLE_DEVICES=1 isaaclab -p scripts/PhyGS/examples/replay_plans.py \
        scripts/PhyGS/output_check/<YYYYMMDD-HHMM>_house_drawer [--passes 2] [--output plans.json]

Prints each plan's outcome as recorded and as planned again, with the plan time; ``--passes`` plans
them all again that many times (later passes: a warm planner). ``--output`` writes every result as
JSON. ``--backend`` overrides the recorded backend (``curobo`` or ``moveit``); the settings
(``planner`` in ``world/environment.yaml``, e.g. cuRobo's dependencies) are this checkout's.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]  # The PhyGS checkout.
sys.path.insert(0, str(PROJECT))  # Runs from a checkout without pip install.

from phygs.recording.reader import Recording  # noqa: E402
from phygs.skills.manipulation.execution.arm_planner import worker_command  # noqa: E402
from phygs.skills.manipulation.execution.planners.interface import request_from_json  # noqa: E402
from phygs.skills.manipulation.execution.planning import PlannerProcess  # noqa: E402
from phygs.world.build import SETTINGS, load_settings  # noqa: E402

MODEL_FILES = ("urdf", "moveit_urdf", "srdf")  # The model's entries that name its files.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="A finished run's folder (its episode.rrd)")
    parser.add_argument("--passes", type=int, default=1, help="Plan every request this many times")
    parser.add_argument("--backend", choices=("curobo", "moveit"), default=None, help="Default: the recorded one")
    parser.add_argument("--output", type=Path, default=None, help="Write every result here (JSON)")
    args = parser.parse_args()
    recording = Recording(args.run)
    planner = recording.properties.get("planner")
    if not planner:
        parser.error(f"{args.run}: the run had no arm planner (no 'planner' in its recording)")
    records = recording.plan_requests()
    settings = dict(load_settings(SETTINGS)["planner"], backend=args.backend or planner["backend"])
    with tempfile.TemporaryDirectory(prefix="phygs_replay_plans_") as folder:
        files = Path(folder)
        model = dict(planner["model"])
        for key in MODEL_FILES:  # The model's files, written again here, named as the model names them.
            if key in model:
                path = files / Path(model[key]).name
                path.write_text(planner["files"][path.name])
                model[key] = str(path)
        (files / "depth").mkdir()
        (files / "world").mkdir()
        command, env = worker_command(settings, PROJECT)
        started = time.perf_counter()
        worker = PlannerProcess(command, model, files, env)
        print(f"{settings['backend']} worker ready in {time.perf_counter() - started:.1f} s; {len(records)} plans")
        results = []
        try:
            for run in range(args.passes):
                for index, record in enumerate(records):
                    request = request_from_json(_request(record, files, index))
                    started = time.perf_counter()
                    result = worker.call(request)
                    wall = time.perf_counter() - started
                    entry = dict(
                        run=run,
                        index=index,
                        t=round(record.time, 3),
                        recorded=record.outcome,
                        success=result["success"],
                        reason=result.get("reason"),
                        plan_seconds=result.get("plan_seconds"),
                        wall_seconds=round(wall, 3),
                    )
                    results.append(entry)
                    same = "same" if result["success"] == record.outcome.get("success") else "DIFFERENT"
                    print(
                        f"pass {run} plan {index:2d} at {record.time:7.2f} s: recorded "
                        f"{_outcome(record.outcome)}, now {_outcome(result)} in {wall:.2f} s ({same})"
                    )
        finally:
            worker.close()
    if args.output is not None:
        args.output.write_text(json.dumps(dict(run=str(args.run), plans=results), indent=1) + "\n")
    last = [entry for entry in results if entry["run"] == args.passes - 1]
    differ = sum(entry["success"] != entry["recorded"].get("success") for entry in last)
    print(f"{len(last)} plans, {differ} with another outcome than recorded")
    return 0


def _request(record, files: Path, index: int) -> dict:
    """The record's request with its depth images written as the worker reads them."""
    data = dict(record.request)
    images = []
    for image in data.get("images") or ():
        image = dict(image)
        camera = image.pop("camera")
        path = files / "depth" / f"{index:06d}_{camera}.npy"
        np.save(path, record.depth[camera])
        images.append(dict(path=str(path), **image))
    data["images"] = images or None
    data["world_path"] = str(files / "world" / f"{index:06d}.npy") if images else None
    return data


def _outcome(result: dict) -> str:
    if result.get("success"):
        return f"planned ({result.get('plan_seconds', float('nan')):.2f} s)"
    return f"failed ({result.get('reason')})"


if __name__ == "__main__":
    raise SystemExit(main())
