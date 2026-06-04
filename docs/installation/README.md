# Installation Overview

PhyGS has **two independent install paths**. Choose the one(s) you need:

| You want to… | Follow |
| --- | --- |
| Generate procedural indoor scenes (Infinigen-Indoors fork). | [`scene_generation.md`](./scene_generation.md) |
| Run the Spot benchmark in IsaacLab — locomotion, perception, AO-Grasp, CuRobo. | [`phygs_simulation.md`](./phygs_simulation.md) |
| Do the end-to-end PhyGS workflow (scene → simulator → agent). | Both, in order. |

The two stacks live in different Python environments and **must not share an interpreter**. Scene generation depends on Blender 4.2, `bpy`, and the Infinigen runtime; the simulation stack depends on IsaacSim 5.1.0, IsaacLab 2.3.0, cuRobo (custom build with nvblox), and Torch as bundled by IsaacSim. Their dependency pins conflict.

## Clone the repo

PhyGS uses git submodules for the AO-Grasp + Contact-GraspNet sources and the cuRobo source consumed by the simulation Docker build. Always clone recursively:

```bash
git clone --recursive https://github.com/m-and-m-lab/PhyGS.git
cd PhyGS
```

If you have an existing checkout, initialize submodules with:

```bash
git submodule update --init --recursive
```

Submodules registered at the PhyGS root (see `.gitmodules`):

| Path | Source | Used by |
| --- | --- | --- |
| `phygs_simulation/third_party/ao-grasp` | `m-and-m-lab/ao-grasp` (which itself vendors `stanford-iprl-lab/contact_graspnet`) | AO-Grasp sidecar Dockerfiles (`docker/aograsp/*.Dockerfile`). |
| `phygs_simulation/third_party/curobo` | `m-and-m-lab/curobo` | `docker/Dockerfile.curobofix` (copied into the IsaacLab container during build). |

The `scene_generation/infinigen/` directory carries its own nested `.gitmodules` (OcMesher), which the recursive init handles for you.

## Hardware prerequisites (both components)

- Linux x86_64. (Scene generation has experimental Mac support; the simulation stack is Linux-only.)
- NVIDIA GPU. cuRobo, IsaacSim, and the AO-Grasp inference services all require CUDA.
- Recent NVIDIA driver and the NVIDIA container runtime for Docker GPU passthrough.

## After install

- Scene-generation outputs land under `scene_generation/infinigen/outputs/...`. Move or reference exported USDs from the simulation stack via the `--scene_usd` argument; do not import the Infinigen Python env from the simulation env.
- Simulation-stack entry points are under `phygs_simulation/scripts/` and `phygs_simulation/tests/isaaclab_test/`, launched via `./isaaclab.sh -p ...` from the IsaacLab root inside the container.
