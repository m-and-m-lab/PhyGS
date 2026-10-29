# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ``phygs`` command: one name for the framework's entry points.

    phygs run --spec <episode.yaml> --agent template --output <folder> [--task_loader module:function]
    phygs ui --output <folder> [--spec ...]            # The robot UI (a person as the agent).
    phygs ui replay <recording>                        # Replay a recorded UI episode.
    phygs scenes list | fetch ...                      # Scene releases from the dataset.
    phygs tests ladder --upto N | <group> ...          # The robot tests (tests/robot_tests/run_suite.py).
    phygs version

Each subcommand runs the script it names with the remaining arguments, so ``phygs run --help`` is
``examples/run_world.py --help``. ``python -m phygs`` is the same command without the console script.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "run": ROOT / "examples" / "run_world.py",
    "ui": ROOT / "phygs" / "ui" / "cli.py",
    "ui-replay": ROOT / "phygs" / "ui" / "replay_cli.py",
    "tests": ROOT / "tests" / "robot_tests" / "run_suite.py",
}
MODULES = {"scenes": "phygs.scenes.fetch"}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command == "version":
        from phygs import __version__

        print(__version__)
        return 0
    if command == "ui" and rest[:1] == ["replay"]:
        command, rest = "ui-replay", rest[1:]
    if command in SCRIPTS:
        script = SCRIPTS[command]
        sys.argv = [str(script), *rest]
        runpy.run_path(str(script), run_name="__main__")
        return 0
    if command in MODULES:
        sys.argv = [MODULES[command], *rest]
        runpy.run_module(MODULES[command], run_name="__main__", alter_sys=True)
        return 0
    print(f"phygs: unknown command {command!r}\n\n{__doc__.strip()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
