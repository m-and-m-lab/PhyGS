# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Copy the framework from the lab's internal interactive-search tree into this checkout.

Code is developed and tested in interactive-search (the lab's internal repository) and flows here
by copy: this script takes the framework paths, drops them into PhyGS's layout and rewrites the
import and path names (``interactive_search`` -> ``phygs``, ``skills`` -> ``phygs.skills``, ...).
It never writes to interactive-search. Review ``git diff`` afterwards; the CHANGELOG entry names
the interactive-search commit the copy was taken from.

    python3 tools/sync_from_interactive_search.py --source ../interactive-search            # whole framework
    python3 tools/sync_from_interactive_search.py --source ../interactive-search --only src/interactive_search/episodes
    python3 tools/sync_from_interactive_search.py --source ../interactive-search --dry_run

Never synced (they are PhyGS's own): README.md, docs/README.md, pyproject.toml, CHANGELOG.md,
PLANS.md, AGENTS.md, the Spot assets, and the pinned policy files
(``phygs/robots/spot/locomotion/{env,locomotion}.yaml``, hashed by the footprint profile).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

PHYGS = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "output_check", ".git", ".scene_usd"}
SKIP_SUFFIX = {".pyc"}
NEVER = {
    "phygs/robots/spot/locomotion/env.yaml",
    "phygs/robots/spot/locomotion/locomotion.yaml",
    "phygs/robots/spot/navigation/transit.json",
}
# (interactive-search path, PhyGS path); a directory copies its tree, a file copies itself.
MAPPING = [
    ("src/interactive_search/agents", "phygs/agents"),
    ("src/interactive_search/episodes", "phygs/episodes"),
    ("src/interactive_search/execution", "phygs/execution"),
    ("src/interactive_search/robots", "phygs/robots"),
    ("src/interactive_search/runtime", "phygs/runtime"),
    ("src/interactive_search/simulation", "phygs/simulation"),
    ("src/interactive_search/ui", "phygs/ui"),
    ("src/interactive_search/helpers/observations", "phygs/observations"),
    ("src/interactive_search/helpers", "phygs/tools"),  # Its modules only; observations/ is above.
    ("src/interactive_search/tasks", "tests/robot_tests/tasks"),
    ("scripts/skills", "phygs/skills"),
    ("scripts/helpers/foragebench.py", "phygs/scenes/fetch.py"),
    ("scripts/navigation/nav2.launch.py", "phygs/execution/navigation/launch/nav2.launch.py"),
    ("scripts/tasks/object_search/api.py", "tests/robot_tests/tasks/object_search_api.py"),
    ("scripts/run_episode.py", "examples/run_world.py"),
    ("scripts/ui_episode.py", "phygs/ui/cli.py"),
    ("scripts/ui_replay.py", "phygs/ui/replay_cli.py"),
    ("configs", "phygs/configs"),
    ("tests/robot_tests", "tests/robot_tests"),
    ("tests", "tests"),  # Unit tests (robot_tests above).
    ("docker", "docker"),
    ("docs", "docs"),  # Topic guides; design pages land in docs/design by the rule below.
]
RULES = [
    (r"interactive_search\.helpers\.observations", "phygs.observations"),
    (r"interactive_search/helpers/observations", "phygs/observations"),
    (r"interactive_search\.helpers", "phygs.tools"),
    (r"interactive_search/helpers", "phygs/tools"),
    (r"interactive_search\.tasks", "phygs_robot_tests.tasks"),
    (r"interactive_search/tasks", "tests/robot_tests/tasks"),
    (r"src/interactive_search", "phygs"),
    (r"interactive_search", "phygs"),
    (
        r"/workspace/isaaclab/scripts/interactive-search/spot_model/",
        "/workspace/isaaclab/scripts/PhyGS/phygs/robots/spot/assets/",
    ),
    (r"scripts/interactive-search", "scripts/PhyGS"),
    (r"(?m)^(\s*)from skills\.", r"\1from phygs.skills."),
    (r"(?<![\w.])skills\.(locomotion|navigation|manipulation)\.api", r"phygs.skills.\1.api"),
    (r"scripts/skills/", "phygs/skills/"),
    (r"(?m)^(\s*)from routines\b", r"\1from phygs_robot_tests.routines"),
    (r"(?m)^import routines(\s*(#.*)?)$", r"from phygs_robot_tests import routines\1"),
    (r"(?m)^import episode(\s*(#.*)?)$", r"from phygs_robot_tests.episodes import episode\1"),
    (r"(?m)^import common(\s*(#.*)?)$", r"from phygs_robot_tests import common\1"),
    (r"(?m)^from common import", r"from phygs_robot_tests.common import"),
    (r"from tasks\.object_search\.api", "from phygs_robot_tests.tasks.object_search_api"),
    (r"from helpers\.foragebench", "from phygs.scenes.fetch"),
    (r"scripts/helpers/foragebench\.py", "phygs/scenes/fetch.py"),
    (r"scripts/navigation/nav2\.launch\.py", "phygs/execution/navigation/launch/nav2.launch.py"),
    (r"(?<![\w/])configs/(environment|navigation|ui|robots|scenes)", r"phygs/configs/\1"),
    (r"(\.\./)*\.scene_usd/tools", "scene_generation/usd_tools"),
    (r"scripts/ui_episode\.py", "phygs/ui/cli.py"),
    (r"scripts/ui_replay\.py", "phygs/ui/replay_cli.py"),
    (r"scripts/run_episode\.py", "examples/run_world.py"),
    (
        r"docs/(episode_harness|manipulation_contract|robot_portability|evaluator_and_robot_tests|framework_benchmark_split|ui|visual|archive)/",
        r"docs/design/\1/",
    ),
]
DESIGN_DIRS = (
    "episode_harness",
    "manipulation_contract",
    "robot_portability",
    "evaluator_and_robot_tests",
    "framework_benchmark_split",
    "ui",
    "visual",
    "archive",
)
TEXT = {".py", ".yaml", ".yml", ".md", ".html", ".js", ".json", ".txt", ".rviz", ".css"}


def destination(source_root: Path, path: Path) -> Path | None:
    """Where a source file lands in PhyGS, or None when it is not part of the framework."""
    rel = path.relative_to(source_root)
    if any(part in SKIP_DIRS for part in rel.parts) or path.suffix in SKIP_SUFFIX:
        return None
    for src, dst in MAPPING:
        src_path = Path(src)
        if rel == src_path or (src_path in rel.parents):
            tail = rel.relative_to(src_path) if rel != src_path else Path(path.name)
            if src == "src/interactive_search/helpers" and tail.parts[0] == "observations":
                return None  # Mapped by the observations rule.
            if src == "tests" and tail.parts[:1] == ("robot_tests",):
                return None  # Mapped by the robot_tests rule.
            if src == "docs":
                if tail.parts[0] in DESIGN_DIRS:
                    return PHYGS / "docs" / "design" / tail
                if tail.suffix != ".md" or tail.name in ("README.md",) or tail.parts[0] != tail.name:
                    return None  # Only the topic guides; PhyGS writes its own index.
            if dst == "phygs/configs" and rel.parts[:2] == ("configs", "scenes"):
                return PHYGS / dst / tail
            return PHYGS / dst / tail
    return None


def rewrite(text: str) -> str:
    for pattern, repl in RULES:
        text = re.sub(pattern, repl, text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="The interactive-search checkout")
    parser.add_argument("--only", type=Path, default=None, help="A path inside the source to sync (default: all)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    start = (source / args.only) if args.only else source
    copied, skipped = 0, 0
    for path in sorted(p for p in start.rglob("*") if p.is_file()) if start.is_dir() else [start]:
        target = destination(source, path)
        if target is None:
            continue
        if str(target.relative_to(PHYGS)) in NEVER:
            skipped += 1
            continue
        if args.dry_run:
            print(f"{path.relative_to(source)} -> {target.relative_to(PHYGS)}")
            copied += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in TEXT:
            try:
                target.write_text(rewrite(path.read_text()))
            except UnicodeDecodeError:
                shutil.copy2(path, target)
        else:
            shutil.copy2(path, target)
        copied += 1
    print(f"{'would copy' if args.dry_run else 'copied'} {copied} files; left {skipped} pinned files alone")


if __name__ == "__main__":
    main()
