# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live robot collision-bound projection using registered rigid-link poses."""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np
import torch


class CollisionFootprint:
    """Project conservative local collision bounds through actual articulated link poses.

    This includes enabled collision shapes only, not visual meshes. Local extent
    boxes enclose collision geometry and can overestimate rounded/convex shapes.
    Sampling captures tested motion, not all possible robot configurations. Bounds are taken
    relative to the robot's ``base`` (:class:`~.base.BaseLink`).
    """

    def __init__(self, stage, robot, root_path: str, *, base) -> None:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self._robot, self._base = robot, base
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise ValueError(f"Missing robot prim: {root_path}")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        vertices, owners, manifest = [], [], []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision = UsdPhysics.CollisionAPI(prim)
            if not collision.GetCollisionEnabledAttr().Get():
                continue
            body = prim
            while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body.IsValid() or body.GetName() not in robot.body_names:
                raise ValueError(f"Collision shape has no registered rigid link: {prim.GetPath()}")
            boundable = UsdGeom.Boundable(prim)
            if not boundable:
                raise ValueError(f"Unsupported non-boundable collision shape: {prim.GetPath()}")
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, Usd.TimeCode.Default())
            if extent is None or len(extent) != 2:
                raise ValueError(f"Missing collision extent: {prim.GetPath()}")
            transform = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
            local = [
                list(transform.Transform(Gf.Vec3d(*corner)))
                for corner in itertools.product(*zip(extent[0], extent[1], strict=True))
            ]
            vertices.extend(local)
            owners.extend([robot.body_names.index(body.GetName())] * len(local))
            # Rounded to 1 nm: the link-frame bounds are composed through world transforms, so placing the
            # robot elsewhere changes them by ~1e-16 m and must not change the geometry identity.
            bounds = [[round(value, 9) + 0.0 for value in corner] for corner in local]
            manifest.append(dict(path=str(prim.GetPath()).removeprefix(root_path), body=body.GetName(), bounds=bounds))
        if not vertices:
            raise ValueError("No enabled robot collision geometry found.")
        self._vertices = torch.tensor(vertices, device=robot.device, dtype=torch.float32)
        self._owners = torch.tensor(owners, device=robot.device, dtype=torch.long)
        self.manifest = manifest  # Per collider: path below the robot, owning link, link-frame bounds [m].
        self.geometry_sha256 = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.colliders = len(manifest)
        self.manifest = manifest

    def points(self) -> np.ndarray:
        """Return live XY collision bounds [m] in a level frame at the robot base's yaw."""
        from isaaclab.utils.math import quat_apply

        pose = self._robot.data.body_link_pose_w.torch[0][self._owners]
        world = quat_apply(pose[:, 3:], self._vertices) + pose[:, :3]
        root = self._base.pose()
        x, y, z, w = root[3:].tolist()
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        relative = world[:, :2] - root[:2]
        c, s = math.cos(yaw), math.sin(yaw)
        return (
            torch.stack((c * relative[:, 0] + s * relative[:, 1], -s * relative[:, 0] + c * relative[:, 1]), dim=1)
            .cpu()
            .numpy()
        )
