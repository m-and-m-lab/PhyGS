# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A run's timeline and path, read from its output folder after the run (``trace.json``).

The trace has one lane per actor, each a list of items with a start ``s`` and end ``e`` [s sim]:

- ``Agent``: the agent's named steps (``session.step``, from ``audit.jsonl``); nested steps are
  kept with their ``depth``, the lane shows depth 0.
- ``Base``, ``Arm``, ``Gripper``: the commands the robot ran, from the state log's command spans
  (:mod:`.state_log`), with their final status.
- ``Grip``: both jaws on a body, with the peak squeeze [N] (the report's ``grasp_contacts``).
- ``Bumps``: every other arm contact, with its peak force [N] (the report's ``contacts``).

``holding`` lists the intervals the gripper held an object (from the state log), named by the
body gripped then; ``narration`` the skills' lines (time, kind, text); ``title`` names the run. The
path holds the robot's poses at ``rate``, Nav2's routes and, when the agent read it through a hint,
the target's position; ``z`` is the robot's height at the start [m], which picks the floor plan.
:func:`compose` makes the trace from what a run logs, as it goes or after it; :func:`build` reads a
finished run's folder. Either is one JSON-friendly dict.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from phygs.recording.state_log import States

LANES = ("Agent", "Base", "Arm", "Gripper", "Grip", "Bumps")
GRIPPER = frozenset({"open_gripper", "close_gripper"})
BASE_LANE, ARM_LANE = "base", "arm"  # The command timeline's lanes (recording.timeline).
STEP = "session.step"  # The audit's call name of step markers.


def read_audit(path: Path) -> list[dict]:
    """The audit's lines (``audit.jsonl``); none when the file is missing."""
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def steps(audit: list[dict], end: float) -> list[dict]:
    """The agent's steps from the audit's step markers: label, start, end [s], depth, status
    (``failed`` when the step raised, ``open`` when the run ended inside it)."""
    found, open_steps = [], []
    for line in audit:
        if line.get("call") != STEP:
            continue
        if line.get("event") == "begin":
            item = dict(label=line["name"], s=line["t"], e=None, depth=len(open_steps), status="succeeded")
            open_steps.append(item)
            found.append(item)
        elif line.get("event") == "end" and open_steps:
            item = open_steps.pop()
            item["e"] = line["t"]
            if line.get("error"):
                item.update(status="failed", note=line["error"])
    for item in open_steps:
        item.update(e=end, status="open")
    return found


def command_items(commands: list[dict], end: float) -> dict[str, list[dict]]:
    """The command spans on the Base, Arm and Gripper lanes. A command a skill issued inside another
    (``look_at`` moving the joints, ``move_to_pose`` running its trajectory) is left out: a span that
    lies within another one on its lane, or has the same times as one opened after it (the outer
    command's span opens when its call returns, after the commands it issued)."""
    lanes = {"Base": [], "Arm": [], "Gripper": []}
    for span in commands:
        lane = "Base" if span["lane"] == BASE_LANE else "Gripper" if span["name"] in GRIPPER else "Arm"
        lanes[lane].append(
            dict(
                label=span["name"],
                s=span["start"],
                e=end if span["end"] is None else span["end"],
                status=span["status"],
            )
        )
    for name, items in lanes.items():
        lanes[name] = [item for i, item in enumerate(items) if not _inside_another(i, items)]
    return lanes


def _inside_another(index: int, items: list[dict]) -> bool:
    item = items[index]
    for other, outer in enumerate(items):
        if other == index or not (outer["s"] <= item["s"] and item["e"] <= outer["e"]):
            continue
        if (outer["s"], outer["e"]) != (item["s"], item["e"]) or other > index:
            return True
    return False


def contact_items(report: dict) -> tuple[list[dict], list[dict]]:
    """The Grip lane (grasp contacts merged per body) and the Bumps lane (one item per contact record)."""
    grips: dict[str, dict] = {}
    for record in report.get("grasp_contacts", ()):
        grip = grips.setdefault(record["body"], dict(s=record["first_s"], e=record["last_s"], peak=0.0))
        grip.update(
            s=min(grip["s"], record["first_s"]),
            e=max(grip["e"], record["last_s"]),
            peak=max(grip["peak"], record["peak_n"]),
        )
    grip_items = [
        dict(label=f"grip {body}", s=grip["s"], e=grip["e"], status="grip", body=body, peak=round(grip["peak"], 1))
        for body, grip in sorted(grips.items(), key=lambda entry: entry[1]["s"])
    ]
    bump_items = sorted(
        (
            dict(
                label=f"{record['link']} on {record['body']}",
                s=record["first_s"],
                e=record["last_s"],
                status="touch",
                body=record["body"],
                peak=round(record["peak_n"], 1),
                note=f"during {record.get('step', '')}",
            )
            for record in report.get("contacts", ())
        ),
        key=lambda item: item["s"],
    )
    return grip_items, bump_items


def holding(states: States, grips: list[dict]) -> list[dict]:
    """Intervals [s] the gripper held an object, named by the body gripped at their start (or the
    nearest grip before it); ``object`` when no grip contact names one. One still held at the last
    sample has the status ``open``."""
    intervals, start = [], None
    for time, held in zip(states.time.tolist(), states.holding.tolist()):
        if held and start is None:
            start = time
        elif not held and start is not None:
            intervals.append([start, time, "succeeded"])
            start = None
    if start is not None:
        intervals.append([start, float(states.time[-1]), "open"])
    named = []
    for s, e, status in intervals:
        before = [grip for grip in grips if grip["s"] <= e]
        body = min(before, key=lambda grip: abs(grip["s"] - s))["body"] if before else "object"
        named.append(dict(s=round(s, 3), e=round(e, 3), label=body, status=status))
    return named


def target(audit: list[dict]) -> list[float] | None:
    """The target's centre (x, y) [m] when the agent read it through ``hints.target``; else None."""
    for line in audit:
        if line.get("call") == "hints.target" and isinstance(line.get("result"), dict):
            hint = next(iter(line["result"].values()), None)
            if isinstance(hint, dict) and "center" in hint:
                return [round(float(value), 3) for value in hint["center"][:2]]
    return None


def compose(
    *,
    end: float,
    title: str = "",
    audit: Sequence[dict] = (),
    commands: Sequence[dict] = (),
    contacts: dict | None = None,
    states: States | None = None,
    narration: Sequence[Sequence] = (),
    poses: Sequence[Sequence[float]] = (),
    rate: float = 10.0,
) -> dict:
    """A trace up to ``end`` [s] from what a run logs: the ``audit`` lines (steps, the target hint),
    the command timeline's spans ``commands``, the report's ``contacts`` and ``grasp_contacts``
    (``contacts``), the state log (``states``: holding, the robot's path at ``rate`` [1/s], the
    routes) and the skills' ``narration`` (time, kind, text). Without a state log, ``poses`` (time,
    x, y, yaw) is the path. ``title`` names the run (the episode and its task). A recording composes
    it as the run goes, :func:`build` after it."""
    all_steps = steps(list(audit), end)
    lanes_by_name = command_items(list(commands), end)
    grips, bumps = contact_items(contacts or {})
    lanes = [
        dict(name="Agent", items=[step for step in all_steps if step["depth"] == 0]),
        *(dict(name=name, items=lanes_by_name[name]) for name in ("Base", "Arm", "Gripper")),
        dict(name="Grip", items=grips),
        dict(name="Bumps", items=bumps),
    ]
    routes = []
    if states is not None:
        routes = [
            dict(t=round(route.time, 3), path=[[round(float(x), 2), round(float(y), 2)] for x, y in route.points])
            for route in states.routes
        ]
        poses = states.base_path(rate)
    return dict(
        title=title,
        duration=round(end, 3),
        lanes=lanes,
        steps=all_steps,
        holding=holding(states, grips) if states is not None else [],
        narration=[list(line) for line in narration],
        path=dict(
            rate=rate,
            z=round(float(states.root[0, 2]), 3) if states is not None and len(states) else 0.0,
            poses=[list(pose) for pose in poses],
            routes=routes,
            target=target(audit),
        ),
    )


def build(run: Path, *, rate: float = 10.0) -> dict:
    """The trace and the path of the run in folder ``run`` (``report.json``, ``audit.jsonl``,
    ``state.npz``): ``{title, duration, lanes, steps, holding, narration, path: {rate, z, poses,
    routes, target}}``."""
    run = Path(run)
    report = json.loads((run / "report.json").read_text())
    states = States.load(run)
    task = report.get("task", {})
    return compose(
        end=float(states.time[-1]) if len(states) else float(report.get("sim_seconds", 0.0)),
        title=f"{report.get('spec', {}).get('id', run.name)} | {task.get('goal', '')}: {task.get('description', '')}",
        audit=read_audit(run / "audit.jsonl"),
        commands=states.commands,
        contacts=report,
        states=states,
        narration=report.get("narration", ()),
        rate=rate,
    )
