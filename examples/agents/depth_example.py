# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example agent: walk to the drawer the target is in, look at its handle with the hand camera,
stream RGB-D while the arm moves, and locate the handle in the world from the camera's depth.

It shows the three ways an agent gets data: ``session.observe()`` (the latest packet, time does
not pass), ``session.stream()`` (one packet per camera period while time passes) and a command's
feedback (``session.wait(command)`` blocks until it ends; ``client.feedback(command)`` reads it now).

    ./isaaclab.sh -p scripts/PhyGS/examples/run_world.py \\
        --spec scripts/PhyGS/tests/robot_tests/configs/episodes/dev_house_drawer.yaml \\
        --agent depth_example --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_depth_example
"""

import math

import numpy as np

from phygs.session import PointTarget

HANDLE_AHEAD_M = 1.10  # Stand with the closed handle this far ahead of the base centre [m].


def run(session):
    # Privileged hints, granted by the spec (object_targets, scene_hints); every read is audited.
    target = session.hints.target()  # The box: centre, size, and the drawer it is in.
    drawer = session.hints.part(target.inside)  # The drawer: handle centre, slide direction, opening.
    handle = np.asarray(drawer.handle_center, dtype=float)
    out = np.asarray(drawer.direction[:2], dtype=float)
    out = out / np.linalg.norm(out)  # The direction the drawer opens toward, level.

    # Fold the arm for walking; wait() lets simulated time pass until the command ends.
    feedback = session.wait(session.manipulation.stow())
    print("stow:", feedback.status, feedback.reason)

    # Walk to stand facing the drawer. Nav2 plans on the map it builds from the front cameras' depth.
    x, y = (handle[:2] + out * HANDLE_AHEAD_M).tolist()
    heading = math.atan2(-float(out[1]), -float(out[0]))
    command = session.navigation.navigate_to(x, y, heading, position_tolerance=0.04, heading_tolerance=0.05)
    while session.navigation.feedback(command).status in ("queued", "active", "stopping"):
        session.sleep(0.5)  # Poll instead of wait(): the agent could look at images here meanwhile.
    feedback = session.navigation.feedback(command)
    print("navigate_to:", feedback.status, feedback.reason)

    # Look at the handle. The arm planner plans around obstacles in the robot's own depth images.
    command = session.manipulation.look_at(PointTarget(tuple(drawer.handle_center)), distance_m=0.6, elevation_deg=60.0)
    frames = 0
    for packet in session.stream(until=lambda packet: session.manipulation.feedback(command).status != "active"):
        frames += 1  # One packet per camera period (31 Hz), each with the newest RGB-D of every camera.
    feedback = session.manipulation.feedback(command)
    print(f"look_at: {feedback.status} {feedback.reason} ({frames} packets streamed while the arm moved)")

    # The newest hand camera image: RGB [H,W,3] uint8, depth [H,W] [m] along the optical axis,
    # depth_valid [H,W], intrinsics [3,3] [px], and its capture stamp (tensors; np.asarray reads them).
    packet = session.observe()
    camera = packet.cameras[session.robot.look_camera]  # The camera look_at aims (Spot's "hand").
    rows, cols = camera.depth.shape
    centre = float(camera.depth[rows // 2, cols // 2])
    print(
        f"hand camera {cols}x{rows}, captured at {camera.stamp.time_seconds:.3f} s; depth at the centre {centre:.3f} m"
    )

    # Depth to world: optical-frame points [m], then world-from-optical at the capture time (the
    # world frame is available because the spec grants gt_localization).
    points = np.asarray(camera.points(stride=2))  # Valid pixels only, every 2nd row and column.
    world_from_camera = np.asarray(packet.transform("world", camera.frame, camera.stamp))
    world = points @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    nearest = np.linalg.norm(world - handle, axis=1).min()
    print(f"{len(world)} depth points; nearest to the handle hint: {float(nearest) * 100:.1f} cm")

    feedback = session.wait(session.manipulation.stow())
    print("stow:", feedback.status, feedback.reason)
    session.finish("depth example done")
