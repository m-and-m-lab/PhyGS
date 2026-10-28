# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode specs and agent references; no Isaac Sim imports.

An episode is one self-contained file: the scene (by id, :mod:`.scenes`), where the robot starts and
where objects start, the task block (opaque to the framework; a task object is built from it,
:func:`phygs_robot_tests.tasks.load_task`), which actions the agent may use, what it observes,
which privileged channels it is granted, a simulated-time budget and a seed. The robot tests'
episodes are ``tests/robot_tests/configs/episodes/*.yaml``.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import yaml

from phygs.episodes.scenes import scene_file, valid_id

SCHEMA = 3
# Named action sets (``actions: skills``) -> the RobotControl action interfaces the session holds.
ACTION_SETS = {
    "skills": ("navigation", "manipulation"),
    "motion": ("navigation", "manipulation", "motion"),
    "joints": ("motion", "arm"),
}
UNSUPPORTED_ACTION_SETS = {"whole_body": "leg joint targets need a runtime without the walking policy"}
ACTIONS = ("navigation", "manipulation", "motion", "arm")
GRANTS = ("gt_localization", "known_map", "object_targets", "scene_hints")
AGENTS_PACKAGE = "phygs.agents"


@dataclass(frozen=True)
class ObservationSpec:
    """Robot sensors the agent observes: onboard cameras by name and by role (``camera_roles``: the
    robot's cameras for a role of its ``robot.yaml``, e.g. ``planning``; a spec that names roles runs
    with any robot), joint state and IMU."""

    cameras: tuple[str, ...] = ()
    camera_roles: tuple[str, ...] = ()
    joint_state: bool = True
    imu: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", tuple(self.cameras))
        object.__setattr__(self, "camera_roles", tuple(self.camera_roles))
        if len(set(self.cameras)) != len(self.cameras) or len(set(self.camera_roles)) != len(self.camera_roles):
            raise ValueError("Observed cameras and camera roles must be unique.")

    def for_robot(self, robot) -> tuple[str, ...]:
        """The observed camera names of ``robot`` (a robot description): the named ones, then each
        role's, in order, without repeats."""
        roles = [name for role in self.camera_roles for name in robot.cameras_for(role)]
        return tuple(dict.fromkeys((*self.cameras, *roles)))


@dataclass(frozen=True)
class EpisodeSpec:
    """A validated episode spec; ``actions`` is a named set's actions or the spec's own list."""

    @dataclass(frozen=True)
    class Scene:
        """The scene: its ``id`` (a scene folder, :mod:`.scenes`), a label (``name``, for reports
        and the robot tests' own data; default the id), whether it needs a ground plane and, for a
        fetched release, the ``version`` it pins. The USD file is looked up when first asked for
        (``usd``), so a spec validates without the scene."""

        id: str
        name: str
        ground: bool
        version: str | None = None

        def __post_init__(self) -> None:
            if not valid_id(self.id) or not self.name:
                raise ValueError(f"A scene needs an id (one folder name, no '..') and a name; got {self.id!r}.")
            if type(self.ground) is not bool:
                raise ValueError("The scene's ground is true or false.")
            if self.version is not None and not re.fullmatch(r"\d+\.\d+\.\d+", str(self.version)):
                raise ValueError(f"A scene version is major.minor.patch; got {self.version!r}.")

        @cached_property
        def usd(self) -> Path:
            """The scene's USD file (absolute), from the scene roots."""
            return scene_file(self.id, self.version)

    @dataclass(frozen=True)
    class Robot:
        """Which robot and where it starts: ``model`` names a robot of this repository
        (``phygs/robots/<model>/``) or is the path of a ``robot.yaml`` (absolute,
        resolved against the spec's folder); ``start`` (x, y, heading) [m, m, rad]. ``starts``: the
        start of another robot (by name) whose body needs other room in the scene, used when the
        episode runs with it (:meth:`EpisodeSpec.with_robot`). The base spawns at the robot's own
        height."""

        model: str
        start: tuple[float, float, float]
        starts: dict[str, tuple[float, float, float]] = field(default_factory=dict)

        def __post_init__(self) -> None:
            object.__setattr__(self, "start", tuple(float(value) for value in self.start))
            starts = {str(name): tuple(float(v) for v in start) for name, start in self.starts.items()}
            object.__setattr__(self, "starts", starts)
            for start in (self.start, *starts.values()):
                if not self.model or len(start) != 3 or not all(map(math.isfinite, start)):
                    raise ValueError("The robot is a model and a start [x, y, heading].")

    id: str
    scene: Scene
    robot: Robot
    # Bodies moved before the episode: {name: {position: [x, y, z], frame: body?}} [m], in the world
    # or in another body's authored frame; empty: as the USD file has them.
    objects: dict
    # The third-person video camera {eye, target} [m] (--record), or None.
    record_view: dict | None
    # The task block as written; the framework does not read it (phygs_robot_tests.tasks.load_task does).
    task: dict
    actions: tuple[str, ...]
    observations: ObservationSpec
    grants: tuple[str, ...]
    sim_seconds: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "grants", tuple(self.grants))
        if not self.id:
            raise ValueError("An episode spec needs an id.")
        if not isinstance(self.task, dict):
            raise ValueError("The task is a mapping.")
        for name, where in self.objects.items():
            if set(where) - {"position", "frame"} or len(where.get("position", ())) != 3:
                raise ValueError(f"Object {name!r} needs a position [x, y, z] and at most a frame.")
        if self.record_view is not None and set(self.record_view) != {"eye", "target"}:
            raise ValueError("The record view is {eye, target}.")
        if not self.actions or not set(self.actions) <= set(ACTIONS):
            raise ValueError(f"Actions must be a non-empty subset of {ACTIONS}.")
        if not set(self.grants) <= set(GRANTS) or len(set(self.grants)) != len(self.grants):
            raise ValueError(f"Grants must be unique names from {GRANTS}.")
        if not (math.isfinite(self.sim_seconds) and self.sim_seconds > 0):
            raise ValueError("The budget must be a positive, finite number of simulated seconds.")
        if type(self.seed) is not int:
            raise ValueError("The seed must be an integer.")

    @classmethod
    def load(cls, path: str | Path) -> EpisodeSpec:
        """Read and validate a spec file; a robot.yaml path in it is relative to the file's folder."""
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: an episode spec is a mapping.")
        try:
            return cls.from_dict(data, base=path.resolve().parent)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error

    @classmethod
    def from_dict(cls, data: dict, base: Path | None = None) -> EpisodeSpec:
        """Validate a parsed spec (a relative robot.yaml path is resolved against ``base``, default the
        working folder). ``actions`` is a named set (``skills``, ``motion``, ``joints``) or a list."""
        known = {
            "schema",
            "id",
            "scene",
            "robot",
            "objects",
            "record_view",
            "task",
            "actions",
            "observations",
            "grants",
            "budget",
            "seed",
        }
        if "tier" in data:
            raise ValueError("The field tier was replaced by actions: write actions: skills (or motion, joints).")
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown spec fields {sorted(unknown)}.")
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported spec schema {data.get('schema')!r}; this code reads schema {SCHEMA}.")
        actions = data["actions"]
        if isinstance(actions, str):
            if actions in UNSUPPORTED_ACTION_SETS:
                raise ValueError(f"Actions {actions!r} are not available yet: {UNSUPPORTED_ACTION_SETS[actions]}.")
            if actions not in ACTION_SETS:
                raise ValueError(f"Unknown action set {actions!r}; expected one of {sorted(ACTION_SETS)} or a list.")
            actions = ACTION_SETS[actions]
        budget = data["budget"]
        if set(budget) != {"sim_seconds"}:
            raise ValueError("The budget has one field, sim_seconds.")
        scene = data["scene"]
        if "usd" in scene:
            raise ValueError("Scenes are named by id, not by file: scene: {id: <folder>}; see episodes/scenes.py.")
        if not {"id", "ground"} <= set(scene) or set(scene) - {"id", "name", "ground", "version"}:
            raise ValueError("The scene is {id, ground} and an optional name and version.")
        robot = data["robot"]
        if not {"model", "start"} <= set(robot) <= {"model", "start", "starts"}:
            raise ValueError("The robot is {model, start} and optionally starts {robot: start}.")
        model = str(robot["model"])
        if ".." in Path(model).parts:
            raise ValueError(f"The robot path {model!r} climbs out of the spec's folder ('..').")
        if model.endswith(".yaml") or "/" in model:  # A robot.yaml (or its folder) anywhere: a path.
            model = str(((base or Path.cwd()) / model).resolve())
        return cls(
            id=data["id"],
            scene=cls.Scene(
                scene["id"],
                str(scene.get("name") or scene["id"]),
                scene["ground"],
                None if scene.get("version") is None else str(scene["version"]),
            ),
            robot=cls.Robot(model, tuple(robot["start"]), dict(robot.get("starts") or {})),
            objects=dict(data.get("objects") or {}),
            record_view=data.get("record_view"),
            task=data["task"],
            actions=actions,
            observations=ObservationSpec(**data.get("observations", {})),
            grants=data.get("grants", ()),
            sim_seconds=float(budget["sim_seconds"]),
            seed=data["seed"],
        )

    def with_seed(self, seed: int) -> EpisodeSpec:
        """The same episode with another seed."""
        return replace(self, seed=seed)

    def with_robot(self, name: str, model: str | None = None) -> EpisodeSpec:
        """The same episode with the robot ``name`` (``model``: its name or ``robot.yaml``, default
        the name), from its own entry of ``robot.starts`` if the spec has one, else the start."""
        robot = self.robot
        return replace(self, robot=replace(robot, model=model or name, start=robot.starts.get(name, robot.start)))

    def summary(self) -> dict:
        """The spec as recorded in a report (the task is the task object's to summarize)."""
        return dict(
            id=self.id,
            schema=SCHEMA,
            scene=self.scene.name,
            scene_id=self.scene.id,
            scene_version=self.scene.version,
            scene_usd=str(self.scene.usd),
            robot=self.robot.model,
            robot_start=list(self.robot.start),
            objects=self.objects,
            actions=list(self.actions),
            cameras=list(self.observations.cameras),
            camera_roles=list(self.observations.camera_roles),
            grants=list(self.grants),
            sim_seconds=self.sim_seconds,
            seed=self.seed,
        )


def load_agent(reference: str) -> tuple[str, Callable]:
    """Resolve an agent reference to (name, run function).

    ``reference`` is a module name in :mod:`phygs.agents` (``template``) or
    ``path/to/file.py:function``. The function takes one argument, the session.
    """
    if ":" in reference:
        path, function = reference.rsplit(":", 1)
        path = Path(path).resolve(strict=True)
        module_spec = importlib.util.spec_from_file_location(f"agent_{path.stem}", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        function = "run"
        try:
            module = importlib.import_module(f"{AGENTS_PACKAGE}.{reference}")
        except ModuleNotFoundError as error:
            if error.name != f"{AGENTS_PACKAGE}.{reference}":
                raise
            raise ValueError(f"No agent {reference!r} in {AGENTS_PACKAGE}; give one there or path.py:function.")
    run = getattr(module, function, None)
    if not callable(run):
        raise ValueError(f"Agent {reference!r} has no function {function!r}.")
    return reference, run
