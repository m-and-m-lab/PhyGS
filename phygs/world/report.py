# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The report an episode writes (``report.json``): its schema number, its keys, and :func:`compose`.

A benchmark reads the report by key; the schema number says which keys a PhyGS version promises.
Bump it when a key is added, renamed or removed, and say so in the changelog under "Report". The
world builds its report with :func:`compose`, which refuses a key missing from or absent in
:data:`KEYS`, so the list and the report cannot drift apart.
"""

from __future__ import annotations

REPORT_SCHEMA = 1
KEYS = (
    "schemas",  # {spec, report}: the schema numbers this report follows.
    "spec",  # The episode spec as run (scene id and version, robot, start, objects, actions, grants, budget, seed).
    "task",  # The task's public part: goal and description.
    "provenance",  # Hashes of the robot USD, policy files, scene USD, evaluator thresholds; the code version.
    "agent",  # The agent's name.
    "clock",  # charged | realtime.
    "sim_seconds",  # Simulated time the episode used [s].
    "wall_seconds",  # Wall time [s].
    "ended_by",  # finish, agent_returned/stopped/failed/error, budget, runtime_*, environment_error.
    "reason",  # The agent's finish reason, or what ended it.
    "agent_outcome",  # {outcome, detail} of the agent thread.
    "stopped",  # What the end-of-episode stop cancelled, how long it took, whether it timed out.
    "evaluation",  # The task's result: outcome, reason, checks, measured, conduct, clean, thresholds.
    "usage",  # Session calls per client, privileged reads, levels used.
    "implicit_grants",  # Grants used without a call naming them.
    "commands",  # Every command issued, with its terminal feedback.
    "contacts",  # Unintended arm contacts (link, body, peak N, the command running).
    "grasp_contacts",  # The squeeze on grasped bodies, kept apart.
    "planner",  # Arm planner backend and its startup/warmup times, or None.
    "narration",  # The skills' step lines.
    "runtime",  # The runtime's own result: ticks, events, status.
)
# Grants a client uses without a call naming them (the report lists them as implicit).
IMPLICIT = {
    "gt_localization": "navigation and skills localize with ground truth",
    "known_map": "navigation and the arm planner avoid the scene's known boxes (without it, Nav2 maps the"
    " navigation cameras' depth and the arm plans on depth)",
}


def implicit_grants(grants: tuple[str, ...]) -> dict[str, str]:
    """The granted channels a client reads without a call naming them, with what each one does."""
    return {grant: IMPLICIT[grant] for grant in grants if grant in IMPLICIT}


def compose(**fields) -> dict:
    """The report from its fields, in :data:`KEYS` order; every key and no other."""
    missing, unknown = set(KEYS) - set(fields), set(fields) - set(KEYS)
    if missing or unknown:
        raise ValueError(f"Report keys: missing {sorted(missing)}, not in KEYS {sorted(unknown)} (report schema).")
    return {key: fields[key] for key in KEYS}
