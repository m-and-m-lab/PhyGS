# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start and result of every robot test.

``parser()`` gives the shared flags (``--output``, ``--record``, ``--robot``, the app flags;
``--manual`` where a test has keyboard control). ``Check(name, args)`` starts the app in a fresh
output folder; ``check.record(control, robot)`` films a component test with ``--record`` (episode tests record
through the harness). Component tests build the robot (``--robot``: :func:`robot`) with
:func:`robot_scene` and read keys with :class:`Keyboard` under ``--manual``. ``check.run(body)``
runs the test: it writes ``check.json`` (the test's ``data`` with ``test``, ``status``, ``error``
and ``checks``), prints ``<test>: <status>; <folder>`` and exits 0 only when the test passed (or a
manual run finished). A traceback is printed before the app closes, and a hang leaves every thread's
stack in ``stacks.txt`` (``kill -USR1 <pid>``).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import signal
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]  # The PhyGS checkout (tests/robot_tests is the package phygs_robot_tests).
CONFIGS = ROOT / "configs"
ENVIRONMENT_CONFIG = PROJECT / "phygs" / "configs" / "environment.yaml"
sys.path[:0] = [str(ROOT), str(PROJECT)]  # Runs from a checkout without pip install.


def _environment() -> dict:
    import yaml

    return yaml.safe_load(ENVIRONMENT_CONFIG.read_text())


PHYSICS_DT = float(_environment()["physics_dt"])  # The project's physics step [s] (250 Hz).


class CheckFailed(AssertionError):
    """A robot test's check failed."""


def require(ok: bool, message: str) -> None:
    """Raise :class:`CheckFailed` with ``message`` unless ``ok``."""
    if not ok:
        raise CheckFailed(message)


def robot(args: argparse.Namespace, default: str = "spot"):
    """The robot under test (``--robot``: a name or a ``robot.yaml``; else ``default``), its
    description checked against its URDF. Call it after :class:`Check` started the app: numpy loaded
    before Kit starts can crash Kit's startup (its fork joins NumPy's OpenBLAS threads; 2026-10-28)."""
    from phygs.robots.description import load_robot

    description = load_robot(args.robot or default)
    description.validate_urdf()
    return description


def robot_scene(robot, position=None, cameras: tuple[str, ...] = (), imu: bool = False) -> dict:
    """``robots.scene.make_scene_cfg`` arguments for a component test, as episodes build the scene:
    ``robot`` (a description) at ``position`` [m] (default: its spawn height at the origin), its
    ``cameras`` by name (RGB and depth), the IMU with ``imu``, the dome light of
    ``phygs/configs/environment.yaml`` and the project's physics step. A test adds only its own props and
    test sensors (e.g. an inspection camera, to ``cameras``)."""
    from phygs.robots.scene import robot_cameras

    return dict(
        robot=robot,
        physics_dt=PHYSICS_DT,
        light=_environment()["light"],
        position=position,
        cameras=robot_cameras(robot, tuple(cameras)),
        imu=imu,
    )


class Keyboard:
    """Keys in the Kit window for ``--manual``: the last key pressed (:meth:`take`) and the keys held
    now (``held``). The callback only records keys; the test acts on them between ticks."""

    def __init__(self) -> None:
        import carb
        import omni.appwindow

        self.held: set[str] = set()
        self._pending = None
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def take(self) -> str | None:
        """The key pressed since the last call, if any."""
        key, self._pending = self._pending, None
        return key

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def _on_event(self, event, *unused) -> bool:
        import carb

        name = event.input if isinstance(event.input, str) else event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pending = name
            self.held.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.held.discard(name)
        return True


def config_scene(robot, config: dict) -> dict:
    """``robots.scene.make_scene_cfg`` arguments from a component test's own scene settings: the
    ``robot`` (a description) at ``config["robot"]`` ``position`` [m] and ``rotation_wxyz`` (a turn
    about z), the ``scene``'s dome light and the ``sensors`` (render cameras by name, ``imu``)."""
    import math

    w, _, _, z = config["robot"].get("rotation_wxyz", (1.0, 0.0, 0.0, 0.0))
    scene, sensors = config["scene"], config.get("sensors", {})
    return dict(
        robot=robot,
        physics_dt=PHYSICS_DT,
        light=dict(intensity=scene["dome_light_intensity"], color=scene["dome_light_color"]),
        position=config["robot"]["position"],
        heading=2.0 * math.atan2(z, w),
        cameras=dict(sensors.get("cameras", {})),
        imu=bool(sensors.get("imu", False)),
    )


def parser(description: str, manual: bool = False) -> argparse.ArgumentParser:
    """The shared flags: ``--output`` (a fresh folder), ``--manual`` (keyboard control in the Kit
    window instead of the checks) when the test has it, and the app flags (``--viz kit`` to watch)."""
    from isaaclab.app import AppLauncher

    result = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output", type=Path, required=True, help="Fresh folder for check.json and the test's files")
    result.add_argument("--record", action="store_true", help="Video with the command timeline: third_person.mp4")
    result.add_argument(
        "--robot",
        default=None,
        help="The robot (a name in phygs/robots or a robot.yaml); default: the episode spec's,"
        " or spot for a component test",
    )
    if manual:
        result.add_argument("--manual", action="store_true", help="Keyboard control in the Kit window")
    AppLauncher.add_app_launcher_args(result)
    result.set_defaults(device="cpu")
    return result


class Check:
    """One robot test run: the app, the named checks and the test's own data."""

    def __init__(self, name: str, args: argparse.Namespace) -> None:
        if (args.output / "check.json").exists():
            raise SystemExit(f"{args.output} holds a finished check; choose a fresh output folder")
        args.output.mkdir(parents=True, exist_ok=True)
        self.name, self.output, self.checks, self.data = name, args.output, {}, {}
        self._record, self._video, self._panels = args.record, None, None
        if args.record:
            args.enable_cameras = True
        stacks = (args.output / "stacks.txt").open("w")
        faulthandler.enable(file=stacks, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=stacks, all_threads=True)
        if getattr(args, "manual", False):
            args.visualizer = ["kit"]
        args.headless = "kit" not in (args.visualizer or [])
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(args).app

        def uncaught(kind, error, trace):  # E.g. an import that fails after the app started.
            traceback.print_exception(kind, error, trace)
            self.app.close(exit_code=self._finish("failed", error))

        sys.excepthook = uncaught

    def record(self, control, robot, *, eye=(-2.2, -2.2, 2.0), target=(0.6, 0.0, 0.5), period: float = 0.032) -> None:
        """With ``--record``, film ``robot`` (an articulation) every ``period`` [s sim] into
        ``third_person.mp4`` (helpers.video: title, RTF and frame rates, narration, command timeline),
        with the panels (helpers.panels) of what the control observes: the stitched front cameras and
        the hand camera with their depth, Nav2's costmap and, with :meth:`show_planner`, the last arm plan.
        A test without any of these (at its first frame) gets no panels.

        A follow camera: ``eye`` and ``target`` are x, y offsets [m] from the robot's base (world
        axes, smoothed so the gait does not shake the view) and heights [m] above the ground."""
        if not self._record:
            return
        import numpy as np
        import yaml

        from phygs.execution.navigation.footprint import transit_footprint
        from phygs.simulation.base import BaseLink
        from phygs.simulation.cameras import ViewCamera
        from phygs.tools.panels import PanelFeed
        from phygs.tools.video import Video

        camera, self._video = ViewCamera(), Video(self.output / "third_person.mp4", fps=1.0 / period)
        outline = None  # A robot in bring-up has no footprint profile yet (check_footprint measures it).
        if (path := control.robot.navigation.footprint_profile) is not None:
            x0, x1, y0, y1 = transit_footprint(yaml.safe_load(path.read_text())).bounds()
            outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        observes = control.runtime.result()["observations"] is not None  # A test with no sensors has no panels.
        if observes:
            self._panels = PanelFeed(control, footprint=outline)
        base_link = BaseLink(robot, control.robot.base_link)
        every, step = max(1, round(period / control.runtime.physics_dt)), control.step
        eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
        follow = {}

        def recorded() -> None:
            step()
            if control.runtime.tick % every == 0:
                base = np.array([*base_link.pose()[:2].cpu().numpy(), 0.0])
                smooth = follow.get("base")
                follow["base"] = (
                    base if smooth is None or np.linalg.norm(base - smooth) > 2.0 else smooth + 0.15 * (base - smooth)
                )
                sources = self._panels.sources(control.observe()) if observes else None
                if "panels" not in follow:  # A video keeps one frame size: panels only if the test has any.
                    follow["panels"] = observes and (sources.drawable() or self._panels.planner is not None)
                self._video.frame(
                    camera.capture(follow["base"] + eye, follow["base"] + target),
                    title=self.name,
                    sim_time=control.runtime.time_seconds,
                    lines=control.narration.lines(),
                    spans=control.timeline.spans,
                    panels=sources if follow["panels"] else None,
                )

        control.step = recorded

    def show_planner(self, planner) -> None:
        """Show ``planner``'s last plan (an ``ArmPlanner``) in the recorded video; nothing without ``--record``."""
        if self._panels is not None:
            self._panels.planner = planner

    def measure(self, name: str, value, ok: bool, message: str = "") -> None:
        """Record a measured value with its verdict; raise :class:`CheckFailed` when it failed."""
        self.checks[name] = dict(value=value, passed=bool(ok))
        print(f"check {name}: {value} ({'passed' if ok else 'FAILED'})", flush=True)
        if not ok:
            raise CheckFailed(f"{name}: {message} ({value})")

    def run(self, body: Callable[[], str | None]) -> None:
        """Run ``body()`` (it may return ``"manual_finished"``), write the result and close the app."""
        try:
            status, error = body() or "passed", None
        except BaseException as exc:  # noqa: BLE001 - every failure is reported, then the app closes.
            traceback.print_exc()
            status, error = "failed", exc
        self.app.close(exit_code=self._finish(status, error))

    def _finish(self, status: str, error: BaseException | None) -> int:
        if self._video is not None:
            self._video.close()
            self._video = None
        error = None if error is None else f"{type(error).__name__}: {error}"
        record = {**self.data, "test": self.name, "status": status, "error": error, "checks": self.checks}
        (self.output / "check.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(f"{self.name}: {status}; {self.output}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return 0 if status in ("passed", "manual_finished") else 1
