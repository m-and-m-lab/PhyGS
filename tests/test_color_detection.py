# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from phygs.observations.color_detection import green_pixel, magenta_pixel


def test_magenta_evidence_ignores_green_distractor_and_invalid_depth():
    rgb = torch.zeros((10, 10, 3), dtype=torch.uint8)
    rgb[1:4, 1:4, 1] = 200
    rgb[6:9, 6:9, 0] = 200
    rgb[6:9, 6:9, 2] = 200
    camera = SimpleNamespace(rgb=rgb, depth_valid=torch.ones((10, 10), dtype=torch.bool))
    assert green_pixel(camera) == (2, 2)
    assert magenta_pixel(camera) == (7, 7)
    camera.depth_valid[6:9, 6:9] = False
    assert magenta_pixel(camera) is None
