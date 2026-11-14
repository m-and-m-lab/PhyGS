from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]  # The PhyGS checkout: phygs/ lives here.
# The robot tests are the package phygs_robot_tests (tests/robot_tests, mapped by pyproject); the editable
# install provides it. Episode plans are importable by file name for the unit tests.
for path in (REPO_ROOT / "tests" / "robot_tests" / "episodes", REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# GitHub's runners have neither Isaac Lab nor the robots' fetched assets. With PHYGS_SKIP_MISSING_ENV=1 (set
# only by .github/workflows/tests.yml) a test that fails for that reason skips instead, and `-rs` lists it.
# Without the variable (the container, any local run) these hooks do nothing: every failure stays a failure.
SKIP_MISSING_ENV = os.environ.get("PHYGS_SKIP_MISSING_ENV") == "1"
ISAAC_MODULES = ("isaaclab", "omni", "isaacsim", "carb")
NOT_INSTALLED = re.compile(rf"No module named '({'|'.join(ISAAC_MODULES)})[.']")


def missing_environment(error: BaseException | None) -> str | None:
    """Why ``error`` is a missing environment rather than a failure (None when it is a failure)."""
    if isinstance(error, ModuleNotFoundError) and (error.name or "").split(".")[0] in ISAAC_MODULES:
        return f"needs Isaac Lab ({error.name} is not installed)"
    if isinstance(error, FileNotFoundError) and "Fetch the robot's assets first" in str(error):
        return "needs the robot's fetched assets"
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if SKIP_MISSING_ENV and report.failed and call.when in ("setup", "call") and call.excinfo is not None:
        reason = missing_environment(call.excinfo.value)
        if reason:
            path, line, _ = item.reportinfo()
            report.outcome, report.longrepr = "skipped", (str(path), (line or 0) + 1, f"Skipped: {reason}")
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    report = yield
    found = NOT_INSTALLED.search(str(report.longrepr)) if SKIP_MISSING_ENV and report.failed else None
    if found:
        report.outcome = "skipped"
        report.longrepr = (str(collector.path), None, f"Skipped: needs Isaac Lab ({found.group(1)} is not installed)")
    return report
