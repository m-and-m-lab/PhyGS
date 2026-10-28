# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Localization provider boundary; ground truth is an explicit experimental mode."""

from typing import Protocol

import torch

from phygs.observations.types import (
    LocalizationObservation,
    ObservationAccess,
    ObservationPacket,
    Stamp,
    Transform,
)


class LocalizationProvider(Protocol):
    """Translate trusted backend measurements into a declared localization product."""

    def observe(
        self, stamp: Stamp, world_from_body: torch.Tensor, velocity_body: torch.Tensor
    ) -> LocalizationObservation:
        """Return localization from pose [m, homogeneous matrix] and velocity [m/s]."""
        ...


class GroundTruthLocalization:
    """Identity world reference, explicitly exposing privileged simulation localization."""

    def observe(
        self, stamp: Stamp, world_from_body: torch.Tensor, velocity_body: torch.Tensor
    ) -> LocalizationObservation:
        """Return exact world pose [m] and body velocity [m/s], marked privileged."""
        return LocalizationObservation(
            stamp,
            Transform("world", "body", stamp, world_from_body.clone(), "ground_truth_localization"),
            velocity_body.clone(),
            bool(torch.isfinite(world_from_body).all() and torch.isfinite(velocity_body).all()),
            "ground_truth",
        )


class LocalizationEstimator(Protocol):
    """Optional sensor-based estimator; no simulator state is supplied.

    ``localize`` must return a pose at the requested capture time [s] or fail.
    Implementations retain the history needed by the configured camera rates.
    """

    name: str
    inputs: ObservationAccess
    reference_frame: str

    def reset(self, episode_id: int) -> None: ...
    def update(self, packet: ObservationPacket) -> None: ...
    def localize(self, stamp: Stamp) -> LocalizationObservation: ...
