# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building a world from its spec: the settings, the combinations refused up front, the cameras to
capture, who observes what, and the third-person view.

Pure functions of the spec, the robot's description and the settings (no Isaac Sim imports), so
they are unit-tested without a simulator; :class:`~phygs.world.World` calls them while it builds
the scene.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from phygs.execution.navigation.footprint import transit_footprint
from phygs.observations.types import CameraPermission, ObservationAccess, ObservationConfig
from phygs.recording.panels import PanelFeed
from phygs.robots.description import RobotDescription
from phygs.runtime.pipeline import PipelineConfig, ServiceDeclaration
from phygs.world.spec import EpisodeSpec

PROJECT = Path(__file__).resolve().parents[2]  # The PhyGS checkout.
SETTINGS = PROJECT / "phygs" / "configs" / "environment.yaml"  # The framework's settings.


def load_settings(path: Path, overrides: dict | None = None) -> dict:
    """The settings file with ``overrides`` applied: a top-level key replaces, a mapping under a
    mapping key merges one level (``{"planner": {"backend": "moveit"}}`` changes only the backend)."""
    settings = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key] = {**settings[key], **value}
        else:
            settings[key] = value
    return settings


def check_supported(spec: EpisodeSpec, robot: RobotDescription) -> None:
    """Fail fast on combinations the framework cannot build yet (no silent substitutes)."""
    unknown = set(spec.observations.cameras) - set(robot.cameras)
    unknown |= {role for role in spec.observations.camera_roles if not robot.cameras_for(role)}
    if unknown:
        raise ValueError(f"Unknown cameras {sorted(unknown)}; {robot.name} has {sorted(robot.cameras)}.")
    grants = set(spec.grants)
    if "navigation" in spec.actions:
        if "gt_localization" not in grants:
            raise NotImplementedError(
                "Navigation without the gt_localization grant needs a localization estimator, not wired into"
                " episodes yet."
            )
        if "known_map" not in grants and not robot.cameras_for("navigation"):
            raise ValueError(f"Navigation without known_map maps depth, and {robot.name} has no navigation camera.")
    if "manipulation" in spec.actions:
        if "gt_localization" not in grants:
            raise NotImplementedError(
                "Manipulation without the gt_localization grant needs a localization estimator, not wired into"
                " episodes yet."
            )
        if robot.look_camera is None or ("known_map" not in grants and not robot.cameras_for("planning")):
            raise ValueError(
                f"Manipulation needs a look camera and, without known_map, planning cameras; {robot.name} lacks them."
            )


def captured_cameras(
    spec: EpisodeSpec,
    robot: RobotDescription,
    observed: tuple[str, ...],
    *,
    viewer_cameras: tuple[str, ...] | None = None,
    record: bool = False,
) -> tuple[str, ...]:
    """The cameras the simulation renders: the agent's (``observed``), plus the skills' look camera
    and, without known_map, the arm planner's and navigation's; when recording, also the video
    panels' (the robot's front and look cameras); with a viewer, also the viewer's."""
    known_map = "known_map" in spec.grants
    cameras = list(observed)
    if viewer_cameras is not None:
        unknown = set(viewer_cameras) - set(robot.cameras)
        if unknown:
            raise ValueError(f"The viewer asks for unknown cameras {sorted(unknown)}")
        cameras += [name for name in viewer_cameras if name not in cameras]
    if record:
        cameras += [name for name in PanelFeed.cameras(robot) if name not in cameras]
    if "manipulation" in spec.actions:
        planner = () if known_map else robot.cameras_for("planning")
        cameras += [name for name in (robot.look_camera, *planner) if name not in cameras]
    if "navigation" in spec.actions and not known_map:
        cameras += [name for name in robot.cameras_for("navigation") if name not in cameras]
    return tuple(cameras)


def observation_pipeline(
    spec: EpisodeSpec,
    robot: RobotDescription,
    observed: tuple[str, ...],
    captured: tuple[str, ...],
    *,
    dt: float,
    period: float,
    record: bool = False,
    viewer_cameras: tuple[str, ...] | None = None,
) -> tuple[PipelineConfig, tuple[ServiceDeclaration, ...]]:
    """What is captured (every ``dt`` [s] for state, every ``period`` [s] for cameras), what the
    agent may read (the spec's observations and grants), and the other consumers: the skills (joints,
    the base pose, every captured camera), the viewer and the video (the cameras each shows)."""
    observed_permissions = tuple(CameraPermission(name) for name in observed)
    captured_permissions = tuple(CameraPermission(name) for name in captured)
    third_person = (CameraPermission("third_person", depth=False),) if record else ()
    capture = ObservationConfig(
        joint_state=True,
        imu=spec.observations.imu,
        localization="ground_truth",
        state_period=dt,
        camera_period=period,
        cameras=captured_permissions + third_person,
    )
    agent = ObservationAccess(
        joint_state=spec.observations.joint_state,
        imu=spec.observations.imu,
        localization="ground_truth" if "gt_localization" in spec.grants else "none",
        cameras=observed_permissions,
    )
    consumers = [
        # The skills read joints, the base pose and their camera, whatever the agent observes.
        ServiceDeclaration(
            "skills", ObservationAccess(joint_state=True, localization="ground_truth", cameras=captured_permissions)
        ),
    ]
    if viewer_cameras is not None:  # A person watching: the viewer's cameras, the joints and the base pose.
        shown = tuple(item for item in captured_permissions if item.name in viewer_cameras)
        access = ObservationAccess(joint_state=True, localization="ground_truth", cameras=shown)
        consumers.append(ServiceDeclaration("viewer", access))
    if record:  # The video and its panels: the cameras it shows and the base pose (recording.panels).
        panel_cameras = PanelFeed.cameras(robot)
        shown = tuple(item for item in captured_permissions if item.name in panel_cameras)
        access = ObservationAccess(localization="ground_truth", cameras=third_person + shown)
        consumers.append(ServiceDeclaration("recorder", access))
    known_boxes = "navigation" in spec.actions and "known_map" in spec.grants
    privileged_sources = ("privileged_known_box_geometry",) if known_boxes else ()
    config = PipelineConfig(capture, agent, ObservationAccess.from_config(capture), privileged_sources)
    return config, tuple(consumers)


def record_view(spec: EpisodeSpec) -> tuple[np.ndarray, np.ndarray]:
    """The third-person camera (eye, target) [m]: the spec's, else behind and above the robot's
    start, looking ahead of it."""
    if spec.record_view is not None:
        return tuple(np.array(spec.record_view[key], dtype=float) for key in ("eye", "target"))
    x, y, heading = spec.robot.start
    c, s = math.cos(heading), math.sin(heading)
    eye = np.array([x - 2.5 * c + 1.5 * s, y - 2.5 * s - 1.5 * c, 2.3])
    return eye, np.array([x + 1.0 * c, y + 1.0 * s, 0.4])


def third_person_camera(record: dict, view: tuple[np.ndarray, np.ndarray]) -> dict:
    """The video's camera config: the settings' ``record`` block (resolution) at ``view``."""
    eye, target = view
    return dict(
        prim_path="{ENV_REGEX_NS}/ThirdPerson",
        **record,
        data_types=["rgb"],
        offset=dict(pos=eye.tolist(), rot=look_rotation(eye, target), convention="world"),
        pinhole=dict(focal_length=20.0, horizontal_aperture=20.955, clipping_range=[0.05, 30.0]),
    )


def look_rotation(eye: np.ndarray, target: np.ndarray) -> list[float]:
    """Camera orientation [qx, qy, qz, qw] (world convention: x forward) looking from eye at target."""
    from scipy.spatial.transform import Rotation

    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    return Rotation.from_matrix(np.column_stack((forward, left, np.cross(forward, left)))).as_quat().tolist()


def transit_outline(robot: RobotDescription) -> np.ndarray | None:
    """The measured transit footprint's rectangle (4x2, body frame [m]) the navigation panel draws,
    or None before the footprint is measured."""
    if robot.navigation.footprint_profile is None:
        return None
    profile = yaml.safe_load(robot.navigation.footprint_profile.read_text())
    x0, x1, y0, y1 = transit_footprint(profile).bounds()
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
