# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image encoding for the UI: JPEG frames and the occupancy map as PNG (depth colours:
:func:`phygs.observations.front_view.depth_colors`).

Binary WebSocket messages are ``[4-byte big-endian header length][JSON header][payload]``.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
from PIL import Image

# Occupancy cells (-1 unknown, 0 free, 100 occupied) as grey levels; free cells where the robot's
# centre cannot stand (its footprint fits at no planning heading) tinted red.
UNKNOWN_GREY, FREE_GREY, OCCUPIED_GREY = 200, 255, 40
TOO_CLOSE = (250, 205, 205)


def jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    """JPEG bytes of an RGB uint8 [H,W,3] image."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def map_png(cells: np.ndarray, blocked: np.ndarray | None = None) -> bytes:
    """PNG of an occupancy grid [y,x] (row 0 at the lowest y), north (+y) up.

    ``blocked`` [y,x]: cells the navigator will not put the robot's centre in; free ones are tinted.
    """
    rgb = np.full((*cells.shape, 3), UNKNOWN_GREY, dtype=np.uint8)
    rgb[cells == 0] = FREE_GREY
    if blocked is not None:
        rgb[(cells == 0) & blocked] = TOO_CLOSE
    rgb[cells >= 50] = OCCUPIED_GREY
    buffer = io.BytesIO()
    Image.fromarray(np.flipud(rgb)).save(buffer, format="PNG")
    return buffer.getvalue()


def binary_message(header: dict, payload: bytes) -> bytes:
    """One binary WebSocket message: header length, JSON header, payload."""
    text = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(text)) + text + payload


def parse_binary_message(message: bytes) -> tuple[dict, bytes]:
    """The header and payload of a binary message (the inverse of :func:`binary_message`)."""
    (length,) = struct.unpack(">I", message[:4])
    return json.loads(message[4 : 4 + length]), message[4 + length :]
