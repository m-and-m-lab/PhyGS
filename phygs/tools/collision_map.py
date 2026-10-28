# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Body-frame boxes for manipulation planning."""

import numpy as np


def boxes_in_body(boxes: list[dict], world_from_body: np.ndarray) -> list[dict]:
    """Transform world-frame boxes into the planner body frame; pose quaternions are xyzw."""
    from scipy.spatial.transform import Rotation

    inverse = np.linalg.inv(world_from_body)
    result = []
    for box in boxes:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(box["pose"][3:]).as_matrix()
        pose[:3, 3] = box["pose"][:3]
        pose = inverse @ pose
        result.append(
            dict(size=box["size"], pose=[*pose[:3, 3].tolist(), *Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()])
        )
    return result
