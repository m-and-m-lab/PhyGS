# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The GPU Isaac Sim renders on (Kit's ``/renderer/activeGpu``). The simulation's other GPU work, the
camera readers and the arm planner's worker, goes to the same GPU, so a run can leave another GPU
free, for example for a model server:

    ./isaaclab.sh -p ... --kit_args="--/renderer/activeGpu=1 --/physics/cudaDevice=1 --/renderer/multiGpu/enabled=false"

Physics stays where the run puts it (the CPU by default). Showing the process only one GPU
(``CUDA_VISIBLE_DEVICES``) does not work: the renderer then finds no device (2026-10-31).
"""

from __future__ import annotations


def render_gpu() -> int | None:
    """The renderer's GPU index (nvidia-smi's order), or None without a running Kit app."""
    try:
        import carb

        value = carb.settings.get_settings().get("/renderer/activeGpu")
    except Exception:  # noqa: BLE001 - no Kit app (unit tests, a planner worker).
        return None
    return None if value is None else int(value)
