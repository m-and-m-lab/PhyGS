# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local browser UI for a person to run Spot through the public skill APIs (``docs/ui.md``).

The simulation runs headless. :class:`~phygs.ui.session.UiSession` runs in the
runner thread and turns the agent view of the observation pipeline into plain data;
:class:`~phygs.ui.server.UiServer` serves the page and a WebSocket from its
own thread. The two exchange data only through a :class:`~phygs.ui.bridge.Bridge`.
"""
