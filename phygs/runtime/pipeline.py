# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit observation views and optional processing, owned by the episode lifecycle."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Generic, Protocol, TypeVar

import torch

from phygs.observations.localization import LocalizationEstimator
from phygs.observations.types import ObservationAccess, ObservationConfig, ObservationPacket
from phygs.runtime.episode import ObservationSource

T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Capture configuration, agent selection, and task-wide information permissions.

    Permissions cover assistance used inside services as well as direct agent
    inputs. Privileged sources name additional trusted providers such as known maps.
    Capture periods [s] remain in :class:`ObservationConfig`.
    """

    capture: ObservationConfig
    agent: ObservationAccess
    permissions: ObservationAccess
    privileged_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "privileged_sources", tuple(self.privileged_sources))

    @classmethod
    def observations_only(cls, capture: ObservationConfig) -> PipelineConfig:
        """Capture robot sensors without supplying localization or derived products."""
        capture = replace(capture, localization="none")
        access = ObservationAccess.from_config(capture)
        return cls(capture, access, access)

    @classmethod
    def localized(cls, capture: ObservationConfig) -> PipelineConfig:
        """Supply configured observations and localization; no map service is implied."""
        access = ObservationAccess.from_config(capture)
        return cls(capture, access, access)


@dataclass(frozen=True)
class ServiceDeclaration:
    """Explicit inputs and additional information sources of an optional component."""

    name: str
    inputs: ObservationAccess
    information_sources: tuple[str, ...] = ()
    privileged_sources: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("information_sources", "privileged_sources", "dependencies", "outputs"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.name or self.name == "agent":
            raise ValueError("A service requires a nonempty name other than 'agent'.")


class ObservationProcessor(Protocol[T]):
    """Consume permitted packets and publish a representation chosen by the processor."""

    def reset(self, episode_id: int) -> None: ...
    def integrate(self, packet: ObservationPacket) -> bool: ...
    def snapshot(self) -> T: ...


@dataclass(frozen=True)
class ProcessingService(Generic[T]):
    """Bind a processor to declared inputs; publication to the agent is opt-in."""

    declaration: ServiceDeclaration
    processor: ObservationProcessor[T]
    expose_to_agent: bool = False


class ObservationView:
    """Read-only packet interface supplied to one configured consumer."""

    def __init__(self, pipeline: ObservationPipeline, name: str) -> None:
        self._pipeline, self._name = pipeline, name

    def observe(self) -> ObservationPacket:
        """Read the consumer's detached packet without advancing the clock."""
        return self._pipeline.observe_for(self._name)


class ObservationPipeline:
    """Capture once, process once per tick, and provide independently filtered views.

    Processors are an explicitly ordered list, not an implicit plugin graph.
    Agent-internal mapping and localization require no registered processor.
    """

    def __init__(
        self,
        source: ObservationSource,
        config: PipelineConfig,
        *,
        processors: tuple[ProcessingService, ...] = (),
        consumers: tuple[ServiceDeclaration, ...] = (),
        localization: LocalizationEstimator | None = None,
    ) -> None:
        self.validate(config, processors=processors, consumers=consumers, localization=localization)
        self._source, self._config = source, config
        self._processors, self._localization = tuple(processors), localization
        self._declarations = tuple(service.declaration for service in processors) + consumers
        self._access = {"agent": config.agent, **{d.name: d.inputs for d in self._declarations}}
        self._packet: ObservationPacket | None = None
        self._episode_id: int | None = None
        self._durations = {s.declaration.name: [] for s in processors}

    @staticmethod
    def validate(
        config: PipelineConfig,
        *,
        processors: tuple[ProcessingService, ...] = (),
        consumers: tuple[ServiceDeclaration, ...] = (),
        localization: LocalizationEstimator | None = None,
    ) -> None:
        """Reject unavailable inputs and forbidden indirect assistance before simulation."""
        captured = ObservationAccess.from_config(config.capture)
        captured.require(config.permissions, consumer="capture")
        available = captured
        if localization is not None:
            if captured.localization != "none":
                raise ValueError("Estimated localization requires a capture source without ground-truth localization.")
            if not localization.name or not localization.reference_frame or localization.reference_frame == "body":
                raise ValueError("An estimator requires a name and a fixed reference frame.")
            localization.inputs.require(captured, consumer=localization.name)
            available = replace(captured, localization="estimated")
            available.require(config.permissions, consumer=localization.name)
        config.agent.require(available, consumer="agent")
        names = {"localization"} if localization is not None else set()
        for declaration in (*(service.declaration for service in processors), *consumers):
            if declaration.name in names:
                raise ValueError(f"Duplicate service: {declaration.name}")
            declaration.inputs.require(available, consumer=declaration.name)
            if not set(declaration.privileged_sources) <= set(config.privileged_sources):
                raise PermissionError(f"Forbidden indirect information source in {declaration.name}.")
            if not set(declaration.dependencies) <= names:
                raise ValueError(f"Unresolved or out-of-order dependencies for {declaration.name}.")
            names.add(declaration.name)

    @property
    def cameras(self):
        """The source's render cameras (trusted diagnostics: e.g. a second, restricted view of the
        same render products); agents never hold the pipeline."""
        return self._source.cameras

    @property
    def configuration(self) -> dict:
        """Record direct inputs, service inputs and transitive information provenance."""
        services = []
        sources = {"localization": {self._localization.name, "estimated_localization"}} if self._localization else {}
        for declaration in self._declarations:
            own = set(declaration.information_sources) | set(declaration.privileged_sources)
            if declaration.inputs.localization != "none":
                own.add(declaration.inputs.localization + "_localization")
                if declaration.inputs.localization == "estimated" and self._localization is not None:
                    own.add(self._localization.name)
            for dependency in declaration.dependencies:
                own.update(sources[dependency])
            sources[declaration.name] = own
            services.append({**asdict(declaration), "effective_information_sources": sorted(own)})
        return {
            **asdict(self._config),
            "services": services,
            "agent_products": [s.declaration.name for s in self._processors if s.expose_to_agent],
            "localization_estimator": None
            if self._localization is None
            else {
                "name": self._localization.name,
                "inputs": asdict(self._localization.inputs),
                "reference_frame": self._localization.reference_frame,
            },
        }

    def view(self, name: str) -> ObservationView:
        """Bind a named consumer to its permitted packet interface."""
        if name not in self._access:
            raise KeyError(name)
        return ObservationView(self, name)

    def reset(self, episode_id: int) -> None:
        """Discard derived state before capturing and processing the new episode."""
        self._packet = None
        self._episode_id = episode_id
        self._durations = {s.declaration.name: [] for s in self._processors}
        if self._localization is not None:
            self._localization.reset(episode_id)
        for service in self._processors:
            service.processor.reset(episode_id)
        self._source.reset(episode_id)
        self._consume()

    def update(self, episode_id: int, tick: int) -> None:
        """Update sensors and optional processors at the runtime-owned tick."""
        if episode_id != self._episode_id:
            raise RuntimeError("Reset the pipeline before changing episodes.")
        if self._packet is not None and tick == self._packet.stamp.tick:
            return
        if self._packet is not None and tick < self._packet.stamp.tick:
            raise ValueError("Pipeline ticks must increase.")
        self._source.update(episode_id, tick)
        self._consume()

    def _consume(self) -> None:
        packet = self._source.observe_internal()
        if packet.stamp.episode_id != self._episode_id:
            raise RuntimeError("Capture belongs to another episode.")
        if self._localization is not None:
            estimator = self._localization
            estimator.update(estimator.inputs.select(packet))
            stamps = {packet.stamp, *(camera.stamp for camera in packet.cameras.values())}
            estimates = {}
            for stamp in stamps:
                estimate = estimator.localize(stamp)
                pose = estimate.pose
                if (
                    estimate.stamp != stamp
                    or pose.stamp != stamp
                    or not estimate.valid
                    or estimate.mode != "estimated"
                    or pose.source != "body"
                    or pose.target != estimator.reference_frame
                    or pose.information_source == "ground_truth_localization"
                    or pose.matrix.shape != (4, 4)
                    or not torch.isfinite(pose.matrix).all()
                    or estimate.linear_velocity.shape != (3,)
                    or not torch.isfinite(estimate.linear_velocity).all()
                ):
                    raise ValueError("Estimator returned an invalid pose, frame, provenance or capture time.")
                estimates[stamp] = estimate
            packet = replace(
                packet,
                localization=estimates[packet.stamp],
                transforms=packet.transforms + tuple(value.pose for value in estimates.values()),
            )
        self._packet = packet
        for service in self._processors:
            selected = service.declaration.inputs.select(packet)
            start = time.perf_counter()
            if service.processor.integrate(selected):
                self._durations[service.declaration.name].append(time.perf_counter() - start)

    def observe_for(self, name: str) -> ObservationPacket:
        """Read only the products selected for this consumer."""
        if self._packet is None:
            raise RuntimeError("Reset the pipeline before reading observations.")
        return self._access[name].select(self._packet)

    def observe(self) -> ObservationPacket:
        """Read the agent view, excluding private service inputs."""
        return self.observe_for("agent")

    @property
    def metrics(self) -> dict:
        """Successful processor updates and wall-clock durations [ms] this episode."""
        return {
            name: {"updates": len(values), "mean_ms": 1000 * sum(values) / len(values), "max_ms": 1000 * max(values)}
            for name, values in self._durations.items()
            if values
        }

    def product(self, name: str) -> Any:
        """Read a published processor snapshot in its native representation."""
        if self._packet is None:
            raise RuntimeError("Reset the pipeline before reading products.")
        for service in self._processors:
            if service.declaration.name == name:
                if not service.expose_to_agent:
                    raise PermissionError(f"Service product is private: {name}")
                return copy.deepcopy(service.processor.snapshot())
        raise KeyError(name)
