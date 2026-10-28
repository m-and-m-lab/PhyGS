# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episodes: a spec, the Environment that builds and runs it, and the session an agent gets.

Design: ``docs/design/episode_harness/index.html``. :mod:`.spec` and :mod:`.session` are pure Python;
:mod:`.environment` and the scene builders need a running Isaac Sim app.
"""
