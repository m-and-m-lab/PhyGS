# Docker

PhyGS runs in its own container, built from Isaac Lab's base image plus what the framework uses
(`Dockerfile.phygs`). Isaac Lab main has no ROS 2 image, so this one adds ROS 2.

| Layer | What | Why |
| --- | --- | --- |
| Isaac Lab base image | Isaac Lab main (3.0.0), Isaac Sim 6.2.0, Ubuntu 24.04, Python 3.12 | The simulator |
| ROS 2 Jazzy (`ros-base`), Nav2, `nav2_bringup`, RViz, grid-map, pointcloud-to-laserscan | apt | Navigation: every episode with `navigation`, the robot UI, rungs 6 and up |
| explore_lite (m-explore-ros2 at a pinned commit) | overlay in `/opt/explore_ws`, sourced by every shell | `navigation.explore()`, rung 13 |
| MoveIt (`moveit_py`, OMPL, task constructor, perception) | apt | The optional `--planner moveit` |
| cuRobo V2 (a pinned commit) | `/opt/curobo_deps` (`planner.curobo_dependencies`) | The default arm planner |
| `huggingface_hub`, `pre-commit`, `rerun-sdk`, `imageio-ffmpeg` | pip in Isaac Sim's Python | Scene releases, the commit hooks, run viewing, videos |
| ffmpeg, tmux, Chromium's libraries | apt | Videos, sessions, page screenshots |
| Fast DDS over UDP only (`fastdds_udp.xml`) | `FASTRTPS_DEFAULT_PROFILES_FILE` | No shared-memory transport between containers |

The projects are not in the image: `scripts/` is mounted, and at the first start `phygs_entry.sh`
installs PhyGS, ForageBench and spot_sim_policy (editable, Isaac Sim's Python).

## Build and enter

The container is the profile `ros2` of Isaac Lab's own container script, so it gets Isaac Lab's
volumes, GPUs, host network and naming, and the script's one-time question about X11 forwarding
(windows such as `--viz kit` and RViz on your screen; the answer is kept in `docker/.container.cfg`).

Once, with PhyGS in `IsaacLab/scripts/PhyGS`:

1. Isaac Lab's `.dockerignore` must exclude the projects and data in `scripts/`, or the build copies
   them into the image: `cat scripts/PhyGS/docker/dockerignore.phygs >> .dockerignore` (from `IsaacLab/`).
2. The profile's settings: `cp scripts/PhyGS/docker/env.ros2 docker/.env.ros2`, then set
   `ROS_DOMAIN_ID` in it (one per person or container on the machine: containers share the host
   network).

Then, from `IsaacLab/docker`, with a suffix of your choice (the container is `isaac-lab-ros2-<suffix>`):

```bash
./container.py start ros2 --suffix <suffix> --files ../scripts/PhyGS/docker/docker-compose.phygs.yaml [your.yaml]
./container.py enter ros2 --suffix <suffix>
```

`./container.py config ros2 --suffix <suffix> --files ...` prints the merged configuration without
starting anything. Your own mounts (data that lives on your machine) go in your own file, passed after
PhyGS's; a mount there replaces one with the same container path. The Compose project is
`phygs-<suffix>`, so starting one container never recreates or removes another (Isaac Lab's own
profiles all share the project `docker`, and the script starts with `--remove-orphans`).

The container has an 8 GB `/dev/shm` (Nav2 and the planner workers crash with Docker's 64 MB default).

## In every shell

```bash
cd /workspace/isaaclab
isaaclab -p -m phygs version
```

Then the README's quick start. On the host, `sudo cpupower frequency-set -g performance` is
recommended (`powersave` costs about 20% real-time factor).
