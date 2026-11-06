# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhyGS's packages are layers and imports go downward only (``docs/layout.md``).

:data:`LAYERS` lists them top to bottom. A module may import its own package and any package in a
lower layer, never one beside or above it. ``skills.composite`` is a layer of its own inside
``skills``: the composite skills are agent-side code over the session, so they sit above it, while
the clients' folders beside them sit below it. ``composite`` is the deprecated alias of it.
Imports under ``if TYPE_CHECKING:`` are annotations, not dependencies, and are not checked: a lower
layer is sometimes handed an upper layer's object (a skill backend drives the client it serves) and
names its type.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "phygs"
LAYERS = (
    ("ui",),
    ("world", "skills.composite", "composite"),
    ("session",),
    ("robots",),
    ("skills",),
    ("sim", "recording"),
    ("scene", "runtime"),
    ("observations",),
)
RANK = {package: rank for rank, layer in enumerate(LAYERS) for package in layer}
MODULES = sorted(
    path for path in PACKAGE.rglob("*.py") if len(path.relative_to(PACKAGE).parts) > 1 and "assets" not in path.parts
)


def package_of(parts) -> str:
    """The layered package of a module path or an import's name below ``phygs``."""
    parts = tuple(parts)
    return "skills.composite" if parts[:2] == ("skills", "composite") else parts[0]


LAYERED = [path for path in MODULES if package_of(path.relative_to(PACKAGE).parts) in RANK]


def runtime_imports(path: Path) -> list[tuple[int, str]]:
    """(line, module) of every import outside ``if TYPE_CHECKING:`` blocks; ``from a import b``
    yields ``a.b`` too, which names a module when ``b`` is one."""
    tree = ast.parse(path.read_text())
    typing_only = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)
        for child in ast.walk(node)
    }
    found = []
    for node in ast.walk(tree):
        if id(node) in typing_only:
            continue
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found += [(node.lineno, node.module)] + [(node.lineno, f"{node.module}.{a.name}") for a in node.names]
    return found


def test_every_package_has_a_layer():
    packages = {path.name for path in PACKAGE.iterdir() if (path / "__init__.py").is_file()}
    assert (PACKAGE / "skills" / "composite" / "__init__.py").is_file()
    assert packages == {name for name in RANK if "." not in name}, (
        "every package is in LAYERS (and docs/layout.md), and every layer's package exists"
    )


@pytest.mark.parametrize("path", LAYERED, ids=lambda path: str(path.relative_to(PACKAGE)))
def test_imports_go_downward(path):
    own = package_of(path.relative_to(PACKAGE).parts)
    wrong = []
    for line, name in runtime_imports(path):
        parts = name.split(".")
        if parts[0] != "phygs" or len(parts) < 2:
            continue
        other = package_of(parts[1:])
        if other == own or (own == "composite" and other == "skills.composite"):
            continue  # Its own package; the alias re-exports the moved package.
        if other == "skills" and len(parts) == 2:
            continue  # ``from phygs.skills import composite`` names the package, not the clients.
        if other not in RANK or RANK[other] <= RANK[own]:
            wrong.append(f"line {line}: {name}")
    assert not wrong, f"{own} is in layer {RANK[own]} ({LAYERS[RANK[own]]}); it imports beside or above: {wrong}"
