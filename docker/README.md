# Docker

PhyGS runs in Isaac Lab's container, built from Isaac Lab's base image plus what the framework
uses (`Dockerfile.phygs`):

| Layer | What | Why |
| --- | --- | --- |
| Isaac Lab base image | Isaac Sim 6.0.1, Isaac Lab 3.0.0, Ubuntu 24.04, Python 3.12 | The simulator |
| ROS 2 Jazzy (`ros-base`), Nav2, `nav2_bringup` | apt | Navigation: every episode with `navigation`, the robot UI, rungs 6 and up |
| explore_lite (m-explore-ros2 at a pinned commit) | overlay in `/opt/explore_ws` | `navigation.explore()`, rung 13 |
| MoveIt (`moveit_py`, OMPL) | apt | The optional `--planner moveit` |
| cuRobo V2 (a pinned commit) | `/opt/curobo_deps` (`planner.curobo_dependencies`) | The default arm planner |
| PhyGS (editable), `huggingface_hub` | pip in Isaac Sim's Python | The `phygs` command, scene releases |

Scene generation (`scene_generation/infinigen`) is not in this image; it has its own environment.

## Build and enter

From Isaac Lab's `docker/` folder, with PhyGS cloned recursively into `IsaacLab/scripts/PhyGS`:

```bash
cd IsaacLab/docker
./container.py start ros2 --files ../scripts/PhyGS/docker/docker-compose.phygs.patch.yaml
./container.py enter ros2
```

The patch builds Isaac Lab's `ros2` service from `Dockerfile.phygs` (Isaac Lab's base image is built
first); the image and container keep Isaac Lab's names (`isaac-lab-ros2`). `scripts/` is
bind-mounted, so edits on the host show in the container and the editable PhyGS install follows them.

Isaac Lab copies its whole tree into the image, `scripts/` included: delete or move large
`output_check/` folders before a build, or add `scripts/*/output_check/` to Isaac Lab's
`.dockerignore`.

## In every shell

```bash
cd /workspace/isaaclab
export ROS_DOMAIN_ID=48        # One per person or simulator on the machine.
# /opt/explore_ws/install/setup.bash (ROS 2 Jazzy and explore_lite) is sourced by ~/.bashrc.
./isaaclab.sh -p -m phygs version
```

Then the README's quick start. On the host, `sudo cpupower frequency-set -g performance` is
recommended (`powersave` costs about 20% real-time factor).
