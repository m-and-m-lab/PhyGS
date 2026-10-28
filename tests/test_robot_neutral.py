# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The framework names no robot's parts: Spot's joints, links, cameras and classes appear in code only
in its own folder (phygs/robots/spot/). Comments and docstrings may still cite a
measurement made on Spot. Framework code asks the robot description for them instead
(docs/adding_a_robot.md)."""

import io
import re
import tokenize
from pathlib import Path

from phygs.robots.description import load_robot

PROJECT = Path(__file__).resolve().parents[1]
SPOT = load_robot("spot")
# Names every robot may use: the framework's base frame and canonical tool, and plain words.
GENERIC = {"body", "base_link", "hand", "left", "right", "rear"}


def spot_names() -> re.Pattern:
    joints = [joint.get("name") for joint in SPOT._urdf.findall("joint")]
    links = [link.get("name") for link in SPOT._urdf.findall("link")]
    names = sorted((set(joints) | set(links) | set(SPOT.cameras)) - GENERIC, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b|\bSpot[A-Z]\w*|spot_model|robots/spot/assets")


def code_tokens(path: Path):
    """(line, text) of the file's string and name tokens, without comments and docstrings."""
    previous = tokenize.NEWLINE
    for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        statement = previous in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT)
        if token.type == tokenize.STRING and statement:
            previous = token.type  # A docstring (or a bare string statement).
            continue
        if token.type in (tokenize.STRING, tokenize.NAME):
            yield token.start[0], token.string
        if token.type not in (tokenize.COMMENT, tokenize.NL):
            previous = token.type


def test_no_spot_names_in_framework_code():
    pattern, own = spot_names(), PROJECT / "phygs" / "robots" / "spot"
    found = [
        f"{path.relative_to(PROJECT)}:{line}: {text}"
        for root in (PROJECT / "phygs",)
        for path in sorted(root.rglob("*.py"))
        if not path.is_relative_to(own)
        for line, text in code_tokens(path)
        if pattern.search(text)
    ]
    assert not found, "Spot names in framework code (ask the robot description):\n" + "\n".join(found)
