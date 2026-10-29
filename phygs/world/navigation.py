# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Nav2 navigation for a simulated robot: the measured transit footprint, the map source and the servers.

:func:`nav2_navigation` is what an episode (the World) and the robot UI both build: it checks
the live robot against the measured transit footprint profile, then gives Nav2 either the scene's
known boxes (a static map, the privileged ``known_map`` grant) or the robot's navigation cameras
(Nav2's obstacle layer maps their point clouds on a declared flat floor), and starts Nav2's servers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from phygs.execution.navigation.backends.nav2 import Nav2BackendFactory, Nav2Config
from phygs.execution.navigation.backends.nav2_stack import Nav2Stack
from phygs.execution.navigation.backends.ros2_check import require_ros2
from phygs.execution.navigation.floor import DeclaredFloorMap
from phygs.execution.navigation.footprint import load_footprint, transit_footprint, validate_transit
from phygs.execution.navigation.known_map import BoxGeometry, KnownBoxMap
from phygs.sim.footprint import CollisionFootprint

if TYPE_CHECKING:
    from phygs.robots.description import RobotDescription
    from phygs.sim.base import BaseLink

ROBOT_PATH = "/World/envs/env_0/Robot"


@dataclass
class Nav2Navigation:
    """``RobotControl.create`` arguments (:attr:`control_arguments`), the running servers (close
    :attr:`stack` after use) and the files the base's motion depends on (for provenance)."""

    config: object  # phygs.skills.navigation.api.NavigationConfig
    factory: Nav2BackendFactory
    stack: Nav2Stack
    base_files: dict[str, Path]

    @property
    def control_arguments(self) -> dict:
        return dict(navigation_config=self.config, navigation_backend_factory=self.factory)


def nav2_navigation(
    *,
    stage,
    scene,
    robot: RobotDescription,
    base: BaseLink,
    area: dict,
    dt: float,
    output: Path,
    nav2: dict,
    point_cloud: dict | None = None,
    known_boxes: tuple[BoxGeometry, ...] | None = None,
    start: tuple[float, float] | None = None,
) -> Nav2Navigation:
    """Nav2 on the scene's ``known_boxes`` (static map) or, without them, on the ``robot``'s
    navigation cameras' point clouds within ``area`` ({bounds, resolution, clearance} [m]).
    ``point_cloud``: {height [m], max_depth [m], period [s] (how often each camera's newest depth goes
    to Nav2), pixel_stride (every this many pixel rows and columns), voxel_size [m] (one point per
    voxel), mark_margin (the fraction of image rows at the bottom whose points only clear),
    explored_radius [m] (optional: the map is unknown beyond this disk around the robot's ``start``
    (x, y) [m] until the cameras see it; without it the whole floor is declared free)}. ``robot`` is
    the robot's description: its navigation limits, kinematics and transit footprint profile; ``base``
    reads its base link. Nav2's files (and explore_lite's log) go to ``output``."""
    from phygs.robots.control import base_controller_class
    from phygs.skills.navigation.api import NavigationConfig

    if (known_boxes is None) == (point_cloud is None):
        raise ValueError("Nav2 maps either known boxes or point clouds.")
    if robot.base.kinematics != "holonomic":
        raise NotImplementedError(
            f"{robot.name} has a {robot.base.kinematics} base: navigation's fine approach moves sideways; a"
            " differential base needs turn-drive-turn fine moves (not implemented, docs/adding_a_robot.md)."
        )
    profile = robot.navigation.footprint_profile
    if profile is None:
        raise ValueError(
            f"{robot.name} has no measured transit footprint (navigation.footprint_profile): measure it with"
            " tests/robot_tests/components/check_footprint.py (rung 03) first."
        )
    require_ros2()  # Before the servers start: the bridge runs in this process.
    base_files = base_controller_class(robot).files(robot)
    footprint = load_footprint(profile, robot.usd, base_files)
    geometry = CollisionFootprint(stage, scene["robot"], ROBOT_PATH, base=base)
    if geometry.geometry_sha256 != footprint["geometry_sha256"]:
        output.mkdir(parents=True, exist_ok=True)
        (output / "footprint_colliders.json").write_text(json.dumps(geometry.manifest, indent=1))
        raise ValueError(
            f"Live collision geometry ({geometry.colliders} shapes) does not match the measured footprint profile"
            f" ({footprint['collision_shapes']} shapes); see footprint_colliders.json"
        )
    config = NavigationConfig(**robot.navigation.limits, arm_posture=robot.arm.transit_posture)
    validate_transit(footprint, config, dt)
    settings = Nav2Config(**{"explore_log": str(output / "explore.log"), **nav2})
    common = dict(
        nav2=dict(namespace=settings.namespace, inflation_cost_distance=settings.inflation_cost_distance),
        navigation=dict(robot.navigation.limits),
        base_kinematics=robot.base.kinematics,
    )
    if known_boxes is not None:
        maps = KnownBoxMap(scene, known_boxes, footprint=transit_footprint(footprint), **area)
        experiment = dict(**common, map=dict(area), footprint_profile=str(profile))
        sensor_mode = "occupancy_grid"
    else:
        explored = None
        if point_cloud.get("explored_radius") is not None:
            if start is None:
                raise ValueError("An explored disk needs the robot's start.")
            explored = (*map(float, start), float(point_cloud["explored_radius"]))
        maps = DeclaredFloorMap(
            bounds=tuple(area["bounds"]), resolution=area["resolution"], footprint=transit_footprint(footprint),
            clearance=0.0, explored=explored,
        )  # fmt: skip
        settings = replace(
            settings,
            sensor_cameras=robot.cameras_for("navigation"),
            sensor_period=point_cloud["period"],
            sensor_pixel_stride=point_cloud["pixel_stride"],
            sensor_voxel_size=point_cloud["voxel_size"],
            sensor_mark_margin=point_cloud["mark_margin"],
        )
        mapping = dict(
            cameras=list(robot.cameras_for("navigation")),
            bounds=list(area["bounds"]),
            resolution=area["resolution"],
            height=list(point_cloud["height"]),
            max_depth=point_cloud["max_depth"],
        )
        experiment = dict(**common, mapping=mapping, transit=dict(footprint_profile=str(profile), clearance=0.0))
        sensor_mode = "pointcloud"
    stack = Nav2Stack(experiment, output, sensor_mode=sensor_mode)
    return Nav2Navigation(config, Nav2BackendFactory(settings, maps=maps), stack, base_files)
