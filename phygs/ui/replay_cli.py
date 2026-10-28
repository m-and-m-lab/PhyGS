# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded robot UI episode in the browser, without the simulation (docs/design/ui/design.html).

Give the output folder of ``cli.py`` (or its ``recording`` folder). Open the printed URL;
the page shows what the person saw, read-only, with pause, speed and restart.

    ./isaaclab.sh -p scripts/PhyGS/phygs/ui/replay_cli.py \\
      scripts/PhyGS/output_check/<YYYYMMDD-HHMM>_ui_grid
"""

from __future__ import annotations

import argparse
import secrets
import signal
import sys
import threading
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # The PhyGS checkout.
sys.path.insert(0, str(PROJECT))  # Runs from a checkout without pip install.

from phygs.ui.replay import SPEEDS, ReplayServer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording", type=Path, help="Output folder of cli.py, or its recording folder")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Address to serve on; keep it local and tunnel for remote use"
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--token", default=None, help="Access token (default: a new random one)")
    parser.add_argument(
        "--speed", type=float, default=1.0, choices=SPEEDS, help="Replay speed (the page can change it)"
    )
    args = parser.parse_args()
    folder = args.recording / "recording" if (args.recording / "recording").is_dir() else args.recording
    if not (folder / "messages.jsonl").is_file():
        parser.error(f"No recording in {args.recording}")
    server = ReplayServer(
        folder, host=args.host, port=args.port, token=args.token or secrets.token_urlsafe(9), speed=args.speed
    )
    server.start()
    print(f"\nRobot UI replay: {server.url}\n", flush=True)
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    done.wait()
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
