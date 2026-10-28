# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Private process entrypoint for the two planner experiments."""

import argparse
import json
import os
import sys
import traceback

from .interface import request_from_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("moveit", "curobo"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response_fd", type=int, required=True)
    args = parser.parse_args()
    with os.fdopen(args.response_fd, "w", buffering=1) as output:
        try:
            if args.backend == "moveit":
                from .moveit import Planner
            else:
                from .curobo import Planner
            with open(args.config) as f:
                planner = Planner(json.load(f))
            output.write(json.dumps({"ready": True}) + "\n")
            for line in sys.stdin:
                request = request_from_json(json.loads(line))
                result = getattr(planner, request.kind)(request)
                output.write(json.dumps(result, allow_nan=False) + "\n")
            planner.close()
        except Exception as exc:
            traceback.print_exc()
            output.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
            raise
    # Every response is sent. Skip interpreter teardown: MoveIt (Jazzy) segfaults there while
    # it destroys the Octomap updater.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
