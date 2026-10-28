# Docker

This directory contains the Isaac Lab container patch for cuRobo-compatible dependencies.

## Files

| Path | Purpose |
| --- | --- |
| `docker-compose.curobofix.patch.yaml` | Compose patch used when launching Isaac Lab with the cuRobo-compatible Dockerfile. |
| `Dockerfile.curobofix` | IsaacLab container customization for cuRobo-related dependencies. |

## IsaacLab Container Patch

From the IsaacLab Docker directory:

```bash
cd /workspace/isaaclab/docker
./container.py start base --files ../scripts/PhyGS/docker/docker-compose.curobofix.patch.yaml
./container.py enter base
```

Inside the container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -i
```
