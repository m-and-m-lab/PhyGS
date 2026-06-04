# Policies

This directory stores learned policy checkpoints used by Interactive Search smoke tests and experiments.

## Contents

| Path | Purpose |
| --- | --- |
| `spot_arm/spot_arm_policy.pt` | Spot arm locomotion policy checkpoint used by `spot_locomotion_wasd.py`. |
| `spot_arm_door/best_agent.pt` | Experimental policy checkpoint for door or articulated-object tasks. |

## Configuration

The locomotion smoke test loads policy metadata from YAML outside this directory:

```text
/workspace/isaaclab/scripts/IsaacRobotics/policies/spot_arm/params/env.yaml
```

The smoke-test config is:

```text
tests/isaaclab_test/spot_locomotion_wasd.yaml
```

## Rules

- Keep checkpoint paths explicit in YAML.
- Do not silently swap checkpoints if the configured file is missing.
- Document policy observation/action assumptions when adding new checkpoints.
- Avoid committing large experimental checkpoints unless they are required benchmark artifacts.
