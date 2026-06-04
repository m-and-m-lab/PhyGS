# PhyGS: Physically-Grounded Controllable Scene Generation

[![Paper](https://img.shields.io/badge/paper-CVPR%202026%20MEIS%20Workshop-b31b1b.svg)](#)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-supported-76b900.svg)](https://developer.nvidia.com/isaac/sim)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](LICENSE)

PhyGS is a simulation framework for the controllable generation of photorealistic, building-scale indoor environments with high-fidelity low-level physics, paired with a full agent stack and hardware abstraction layer for the Boston Dynamics Spot with Arm. It is built to generate and evaluate mobile manipulation tasks that are intractable to run at scale in the real world.

<!-- Drop the Figure 1 teaser here once assets are in place
<p align="center">
  <img src="docs/assets/teaser.png" alt="PhyGS generated scenes rendered in IsaacSim and Spot sim-to-real" width="100%">
</p> -->

## Abstract

The pursuit of generalist robot policies requires evaluation spanning variations of visual environments and physical interactions intractable for a robot to execute in the real world. We introduce PhyGS, a simulation framework designed for the controllable generation of photorealistic, building-scale indoor environments with high-fidelity low-level physics that bridges the gap between procedural scene synthesis and robotic control. By extending Infinigen-Indoors within the IsaacSim ecosystem, PhyGS transforms static visual backdrops into fully interactable environments through automated object articulation, the assignment of rigid-body physical properties, and the integration of ray-traced lighting. Unlike recent agent-based generative tools that rely on stochastic language models, PhyGS employs a rule-based approach to ensure the controllability required for rigorous benchmarking. Finally, we provide a hardware abstraction layer via a unified API and USD model for the Boston Dynamics Spot robot, demonstrating a seamless transition between simulated evaluation and physical deployment to help identify critical performance gaps in state-of-the-art generalist robotic agents.

## Authors

Aparajito Saha, Zhen Hao Gan, Jinjia Guo, Jacob Skwirsk, Jeremy Acheampong, Anton Arapin, Chahyon Ku, Yue Hu, Nima Fazeli, and Bernadette Bucher.

University of Michigan, Ann Arbor.

## Note on usage

PhyGS is under active development at the Mapping and Motion Lab, and this version of the repository is a beta release intended for early community testing and usage. Please file a GitHub issue detailing any bugs faced or major feature requests, and the team will respond to and resolve these issues as soon as possible. 

## What's in this repository

PhyGS combines three components that previously lived in separate codebases:

1. **Scene generation and export pipeline** — a fork of Infinigen-Indoors extended to add object articulation and semantic population, and to retain physics (joints, collision meshes, rigid-body properties, lighting) through export from Blender to USD.
2. **Agent skills for Spot with Arm in simulation** — low-level controllers (an RL policy for velocity-driven locomotion; LulaIK + cuRobo for 6-DoF arm kinematics and dynamics) and high-level planners (VLFM for semantic navigation, AOGrasp for grasp synthesis).
3. **IsaacSim deployment** — task definitions, physics configuration, and the Docker environment for running the agent in simulation.

These run in **two separate Python environments** that should not be mixed (see [Installation](#installation)).

## Repository structure

<!-- ```
phygs/
├── scene_generation/       # Component 1 — scene generation (Blender / Infinigen env)
│   ├── infinigen/          # vendored Infinigen-Indoors fork (upstream code)
├── phygs_simulation/       # Component 2 — controllers, planners, Spot model
│   └── phygs_agents/
│       ├── controllers/    # locomotion (RL) + manipulation (LulaIK + cuRobo)
│       ├── planners/       # VLFM, AOGrasp
│       └── robots/         # Spot with Arm USD reference + BD-SDK-mirroring API
├── deployment/       # Component 3 — task/env definitions, physics configs, launch scripts
├── docker/           # container used to run components 2–3
├── assets/           # USD models, meshes, checkpoints (git-lfs / release artifacts)
├── examples/         # end-to-end: generate scene → load in sim → run agent
└── docs/             # installation and usage guides
``` -->

```
phygs/
├── scene_generation/       # Component 1 — scene generation (Blender / Infinigen env)
│   ├── infinigen/          # vendored Infinigen-Indoors fork (upstream code)
├── phygs_simulation/       # Component 2 — controllers, planners, Spot model
├── examples/         # end-to-end: generate scene → load in sim → run agent
└── docs/             # installation and usage guides
```

<!-- > The two environments are intentionally isolated: scene generation depends on the Blender
> Python and Infinigen stack, while the agent and deployment side depend on
> IsaacSim/IsaacLab, cuRobo, and PyTorch. Their dependency sets conflict, so each component
> is installed and run independently. -->

## Installation

PhyGS has two install paths that are independent of each other. The scene generation component can be executed either in a Docker container or a conda environment, while the simulation component must be run inside the base IsaacLab container.

Step-by-step instructions live under [`docs/installation/`](docs/installation/):

| Guide | When to follow it |
| --- | --- |
| [Installation overview](docs/installation/README.md) | Prerequisites, repo clone with submodules, environment-split rationale. Start here. |
| [Scene generation](docs/installation/scene_generation.md) | Installing the Infinigen-Indoors fork in `scene_generation/infinigen/` (conda env, `[sim]` extra). |
| [PhyGS simulation](docs/installation/phygs_simulation.md) | Building the IsaacLab Docker container (IsaacSim 5.1.0 / IsaacLab 2.3.0), AO-Grasp sidecars, smoke tests. |

PhyGS uses git submodules for AO-Grasp + Contact-GraspNet and cuRobo. Clone recursively, or initialize after the fact:

```bash
git clone --recursive https://github.com/m-and-m-lab/PhyGS.git
# Or, on an existing checkout:
git submodule update --init --recursive
```

## Citation

If you use PhyGS in your research, please cite:

```bibtex
@misc{2026phygs,
  title     = {PhyGS: Physically-Grounded Controllable Scene Generation},
  author    = {Saha, Aparajito and Gan, Zhen Hao and Guo, Jinjia and
               Skwirsk, Jacob and Acheampong, Jeremy and Arapin, Anton and
               Ku, Chahyon and Hu, Yue and Fazeli, Nima and Bucher, Bernadette},
  note      = {CVPR Workshop on Multi-Agent Embodied Intelligent Systems},
  year      = {2026}
}
```

## Acknowledgments

PhyGS builds on a number of open-source projects, including: 
[Infinigen / Infinigen-Indoors](https://github.com/princeton-vl/infinigen),
[NVIDIA IsaacSim](https://developer.nvidia.com/isaac/sim),
[IsaacLab](https://github.com/isaac-sim/IsaacLab),
[Objaverse-XL](https://objaverse.allenai.org/),
[cuRobo](https://curobo.org/), 
[VLFM](https://naoki.io/portfolio/vlfm.html), 
[AOGrasp](https://stanford-iprl-lab.github.io/ao-grasp/). 

We thank the authors and maintainers of these tools. Please refer to each project's license and cite them as appropriate.
