# PhyGS Documentation

This directory holds top-level documentation for the PhyGS framework. PhyGS bundles two components that live in two intentionally isolated Python environments:

| Component | Path | Purpose |
| --- | --- | --- |
| Scene generation | `scene_generation/infinigen/` | Procedural indoor scene synthesis (vendored Infinigen-Indoors fork). Runs in a Blender/Infinigen Python environment. |
| Simulation & agents | `phygs_simulation/` | Spot model, AO-Grasp services, CuRobo planning, IsaacLab benchmark scripts, Spot-SDK-shaped skills API. Runs inside the IsaacLab Docker container. |

The two environments **must not be mixed** — their dependency sets conflict. Install and run each independently per the guides below.

## Installation

- [Overview](./installation/README.md) — environment split, prerequisites, and which guide to follow.
- [Scene generation](./installation/scene_generation.md) — Infinigen-Indoors fork. Thin wrapper around the upstream installation procedure with PhyGS-specific notes.
- [PhyGS simulation](./installation/phygs_simulation.md) — IsaacLab container build, submodule init, AO-Grasp sidecar services. Validated against **IsaacSim 5.1.0** and **IsaacLab 2.3.0**.

## Component documentation

Once installed, each component has its own README and architecture notes:

- `scene_generation/infinigen/README.md` and the upstream docs under `scene_generation/infinigen/docs/`.
- `phygs_simulation/README.md`, `phygs_simulation/ARCHITECTURE.md`, `phygs_simulation/AGENTS.md`, and `phygs_simulation/PLANS.md`.
