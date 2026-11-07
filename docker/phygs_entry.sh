#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# PhyGS container start (the image's ENTRYPOINT): install the projects in scripts/ (editable, into
# Isaac Sim's Python, once per container), then run the command (bash by default).
set -u
W=${ISAACLAB_PATH:-/workspace/isaaclab}
log() { echo "phygs_entry: $*" >&2; }

mkdir -p /var/lib/phygs
for project in PhyGS ForageBench spot_sim_policy; do
  dir=$W/scripts/$project
  marker=/var/lib/phygs/installed_$project
  [ -f "$dir/pyproject.toml" ] || [ -f "$dir/setup.py" ] || continue
  [ -f "$marker" ] && continue
  if isaaclab -p -m pip install --quiet --no-deps -e "$dir" > /var/lib/phygs/install_$project.log 2>&1; then
    touch "$marker"
  else
    log "installing $project failed; see /var/lib/phygs/install_$project.log"
  fi
done

exec "$@"
