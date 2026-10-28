# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab deployment wrapper for the Spot flat-terrain locomotion policy (the ``legged_policy``
base of ``robot.yaml``: ``base.policy`` names this class, ``base.config`` its settings file)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml


class _PolicyYamlLoader(yaml.SafeLoader):
    """Safe YAML loader that understands train-time tuple/slice tags."""


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> tuple[object, ...]:
    return tuple(loader.construct_sequence(node))


def _construct_python_slice(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> slice:
    values = loader.construct_sequence(node)
    return slice(*values)


_PolicyYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
_PolicyYamlLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_python_slice)
_PolicyYamlLoader.add_constructor(None, lambda loader, node: None)

# Policy observation terms in order: the walking terms, then optionally the arm's.
_WALK_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)
_ARM_TERMS = ("arm_joint_pos", "arm_joint_vel")


def _load_yaml(path: str | Path) -> dict:
    data = yaml.load(Path(path).read_text(), Loader=_PolicyYamlLoader)
    return {} if data is None else data


class SpotFlatTerrainPolicy:
    """Minimal Isaac Lab runtime wrapper for the exported Spot locomotion policy.

    ``config_path`` is the locomotion settings file (``locomotion/locomotion.yaml``): the policy and
    its training configuration (paths relative to it), the stand drive gains and the stand switch.
    The policy acts at its trained period (training ``sim.dt`` x ``decimation``, 0.02 s). Physics
    may run at another timestep ``physics_dt`` [s] whose multiple is that period; the decimation
    follows from it.
    A policy trained with the arm in the loop also observes the arm joints' positions [rad],
    relative to the training default, and velocities [rad/s] (``arm_joint_pos``, ``arm_joint_vel``).
    """

    def __init__(self, config_path: str | Path, *, physics_dt: float) -> None:
        config_path = Path(config_path)
        locomotion = _load_yaml(config_path)
        self.policy_path = str((config_path.parent / locomotion["policy_path"]).resolve(strict=True))
        self.env_config_path = str((config_path.parent / locomotion["env_config_path"]).resolve(strict=True))

        config = _load_yaml(self.env_config_path)
        action_cfg = config["actions"]["joint_pos"]
        self._training_dt = float(config["sim"]["dt"])
        self._period = self._training_dt * int(config["decimation"])
        self._dt = float(physics_dt)
        self._decimation = round(self._period / self._dt)
        if self._decimation < 1 or not math.isclose(self._decimation * self._dt, self._period, abs_tol=1e-9):
            raise ValueError(f"Physics timestep {self._dt} s does not divide the policy period {self._period} s.")
        self._action_scale = float(action_cfg["scale"])
        self._joint_name_patterns = tuple(str(name) for name in action_cfg["joint_names"])
        self._arm_joint_names, self._arm_training_default = self._arm_observation(config)

        self.robot = None
        self.policy = None
        self._leg_joint_ids: list[int] | None = None
        self._leg_joint_names: tuple[str, ...] | None = None
        self._leg_actuator_names: tuple[str, ...] | None = None
        self._default_leg_pos: torch.Tensor | None = None
        self._walk_actuator_gains: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        self._last_action: torch.Tensor | None = None
        self._cached_action: torch.Tensor | None = None
        self._arm_joint_ids: list[int] | None = None
        self._arm_default: torch.Tensor | None = None
        self._foot_body_ids: list[int] | None = None
        rate = round(1.0 / self._dt)
        standing = locomotion["stand_drive"]
        if rate not in standing:
            raise ValueError(f"No stand drive gains for {rate} Hz physics; rates: {sorted(standing)}.")
        # PhysX joint-drive gains while standing: (Kp [N·m/rad], Kd [N·m·s/rad]) for hips and knees.
        self._stand_drive = {part: tuple(float(v) for v in standing[rate][part]) for part in ("hip", "knee")}
        switch = locomotion["stand_switch"]
        self._stand_settle_time = float(switch["max_seconds"])  # The latest switch after a zero command [s].
        self._still_base_speed = float(switch["base_speed"])  # [m/s]
        self._still_yaw_speed = float(switch["yaw_speed"])  # [rad/s]
        self._still_joint_speed = float(switch["joint_speed"])  # [rad/s]
        self._still_feet_spread = float(switch["feet_spread"])  # [m]
        self._feet_down_after = float(switch["feet_down_after"])  # [s]
        self._feet_down_spread = float(switch["feet_down_spread"])  # [m]
        self._stand_pose: torch.Tensor | None = None  # Leg targets [rad] while standing.
        self._zero_command_time = 0.0
        self._standing = False
        self._policy_counter = 0

    @staticmethod
    def _arm_observation(config: dict) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """Arm joints the policy observes and their training default positions [rad] (none: empty)."""
        terms = config["observations"]["policy"]
        names = tuple(name for name, term in terms.items() if isinstance(term, dict))
        if names == _WALK_TERMS:
            return (), ()
        if names != _WALK_TERMS + _ARM_TERMS:
            raise ValueError(f"Unsupported policy observation terms: {names}.")
        joints = [tuple(terms[term]["params"]["asset_cfg"]["joint_names"]) for term in _ARM_TERMS]
        if joints[0] != joints[1]:
            raise ValueError("The arm position and velocity observations must cover the same joints.")
        defaults = config["scene"]["robot"]["init_state"]["joint_pos"]
        values = []
        for joint in joints[0]:
            matches = [float(v) for pattern, v in defaults.items() if re.fullmatch(pattern, joint)]
            if len(matches) != 1:
                raise ValueError(f"Arm joint {joint} needs exactly one training default position, found {matches}.")
            values.append(matches[0])
        return joints[0], tuple(values)

    @property
    def control_dt(self) -> float:
        return self._dt

    @property
    def training_dt(self) -> float:
        """Physics timestep the policy was trained with [s]."""
        return self._training_dt

    def initialize(self, robot) -> None:
        if robot.data.joint_pos.torch.shape[0] != 1:
            raise ValueError(
                f"SpotFlatTerrainPolicy expects one robot instance, got {robot.data.joint_pos.torch.shape[0]}."
            )

        self.robot = robot
        self.policy = torch.jit.load(self.policy_path, map_location=robot.device).eval()

        leg_joint_ids, leg_joint_names = robot.find_joints(list(self._joint_name_patterns), preserve_order=False)
        if len(leg_joint_ids) != 12:
            raise RuntimeError(
                "Spot locomotion policy expected 12 leg joints, "
                f"resolved {len(leg_joint_ids)}: {tuple(leg_joint_names)}"
            )

        self._leg_joint_ids = list(leg_joint_ids)
        self._leg_joint_names = tuple(leg_joint_names)
        leg_joint_id_set = set(self._leg_joint_ids)
        self._leg_actuator_names = tuple(
            name
            for name, actuator in robot.actuators.items()
            if leg_joint_id_set.intersection(int(joint_id) for joint_id in actuator.joint_indices)
        )
        if not self._leg_actuator_names:
            raise RuntimeError("Spot locomotion policy expected leg actuators, found none.")
        controlled = [int(index) for name in self._leg_actuator_names for index in robot.actuators[name].joint_indices]
        if len(controlled) != 12 or set(controlled) != leg_joint_id_set:
            raise ValueError("Leg actuators must exclusively cover all 12 leg joints.")
        self._default_leg_pos = robot.data.default_joint_pos.torch[:, self._leg_joint_ids].clone()
        if self._arm_joint_names:
            arm_ids, arm_names = robot.find_joints(list(self._arm_joint_names), preserve_order=False)
            if len(arm_ids) != len(self._arm_joint_names):
                raise RuntimeError(
                    f"Spot locomotion policy observes arm joints {self._arm_joint_names}, resolved {arm_names}."
                )
            default = dict(zip(self._arm_joint_names, self._arm_training_default))
            self._arm_joint_ids = list(arm_ids)
            self._arm_default = torch.tensor([[default[name] for name in arm_names]], device=robot.device)
        self._foot_body_ids = [index for index, name in enumerate(robot.body_names) if name.endswith("_foot")]
        if len(self._foot_body_ids) != 4:
            raise RuntimeError(f"The stand switch needs Spot's four feet, found {len(self._foot_body_ids)} '*_foot'.")
        self._walk_actuator_gains = {
            name: (robot.actuators[name].stiffness.clone(), robot.actuators[name].damping.clone())
            for name in self._leg_actuator_names
        }
        self._last_action = torch.zeros((1, len(self._leg_joint_ids)), dtype=torch.float32, device=robot.device)
        self._cached_action = torch.zeros_like(self._last_action)
        self._zero_command_time = 0.0
        self._standing = False
        self._policy_counter = 0

    @property
    def leg_joint_names(self) -> tuple[str, ...]:
        """Policy output order, resolved against the robot's joint order."""
        if self._leg_joint_names is None:
            raise RuntimeError("Initialize the policy before reading joint names.")
        return self._leg_joint_names

    @property
    def inference_dt(self) -> float:
        """Policy inference interval [s]."""
        return self._dt * self._decimation

    @property
    def stand_switch_seconds(self) -> float:
        """The latest stand switch after the command reaches zero [s] (``stand_switch.max_seconds``)."""
        return self._stand_settle_time

    @property
    def standing(self) -> bool:
        return self._standing

    @property
    def stand_pose(self) -> torch.Tensor:
        """Leg targets [rad] held while standing, shape [1, 12] (a copy)."""
        return self._stand_pose.clone()

    def reset(self) -> None:
        """Clear previous action and inference phase without writing robot state."""
        if self._last_action is None or self._cached_action is None:
            raise RuntimeError("Initialize the policy before reset.")
        self._last_action.zero_()
        self._cached_action.zero_()
        self._policy_counter = 0
        self._zero_command_time = 0.0
        self._standing = True
        self._stand_pose = self._default_leg_pos.clone()

    def gain_targets(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return leg stiffness [N·m/rad] and damping [N·m·s/rad]: PhysX joint-drive gains while
        standing, else the walking actuators' explicit-PD gains."""
        if self._standing:
            gains = {j: self._stand_drive["knee" if j.endswith("_knee") else "hip"] for j in self.leg_joint_names}
            return {j: g[0] for j, g in gains.items()}, {j: g[1] for j, g in gains.items()}
        stiffness, damping = {}, {}
        for name in self._leg_actuator_names:
            gains = self._walk_actuator_gains[name]
            for index, joint_id in enumerate(self.robot.actuators[name].joint_indices):
                joint = self.robot.joint_names[int(joint_id)]
                stiffness[joint] = float(gains[0][0, index])
                damping[joint] = float(gains[1][0, index])
        return stiffness, damping

    def compute_targets(self, dt: float, command: Sequence[float] | torch.Tensor) -> torch.Tensor:
        """Compute leg targets [rad] without actuator writes; command is [m/s, m/s, rad/s]."""
        if (
            self.robot is None
            or self.policy is None
            or self._leg_joint_ids is None
            or self._default_leg_pos is None
            or self._leg_actuator_names is None
            or self._walk_actuator_gains is None
            or self._last_action is None
            or self._cached_action is None
        ):
            raise RuntimeError("SpotFlatTerrainPolicy.initialize(robot) must be called before compute_targets().")

        if not math.isclose(dt, self._dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Policy requires physics_dt={self._dt}, got {dt}.")
        if isinstance(command, torch.Tensor):
            command_tensor = command.to(dtype=torch.float32, device=self.robot.device).view(1, 3)
            if not torch.isfinite(command_tensor).all():
                raise ValueError("Velocity command must be finite.")
            zero_command = torch.count_nonzero(command_tensor).item() == 0
        else:
            if len(command) != 3 or not all(math.isfinite(value) for value in command):
                raise ValueError("Velocity command must contain three finite values.")
            zero_command = not any(command)
            command_tensor = None
        if zero_command:
            if self._standing:
                return self._stand_pose.clone()
            self._zero_command_time += float(dt)
            if self._zero_command_time >= self._stand_settle_time or self._still():
                return self._stand_in_place()
        else:
            self._zero_command_time = 0.0

        self._standing = False

        if self._policy_counter % self._decimation == 0:
            if command_tensor is None:
                command_tensor = torch.as_tensor(command, dtype=torch.float32, device=self.robot.device).view(1, 3)
            observation = self._compute_observation(command_tensor)
            with torch.inference_mode():
                action = self.policy(observation).to(dtype=torch.float32)
            if action.shape != self._cached_action.shape or not torch.isfinite(action).all():
                raise ValueError(f"Policy must return finite actions of shape {tuple(self._cached_action.shape)}.")
            self._cached_action.copy_(action)
            self._last_action.copy_(self._cached_action)

        self._policy_counter += 1
        return self._default_leg_pos + self._cached_action * self._action_scale

    def stand_targets(self) -> torch.Tensor:
        """Return default leg positions [rad] and clear policy memory, without robot writes."""
        self.reset()
        return self._default_leg_pos.clone()

    def _still(self) -> bool:
        """Ready to stand: the base has nearly stopped and turns slowly, no leg joint moves fast and
        all feet are down (level within ``feet_spread``); or, from ``feet_down_after`` [s] after the
        zero command, all feet are down within ``feet_down_spread``, whatever the speeds."""
        data = self.robot.data
        feet = data.body_pos_w.torch[0, self._foot_body_ids, 2]
        spread = float(feet.max() - feet.min())
        if self._zero_command_time >= self._feet_down_after and spread < self._feet_down_spread:
            return True
        base = float(torch.linalg.vector_norm(data.root_lin_vel_b.torch[0, :2]))
        yaw = abs(float(data.root_ang_vel_b.torch[0, 2]))
        joints = float(data.joint_vel.torch[0, self._leg_joint_ids].abs().max())
        return (
            base < self._still_base_speed
            and yaw < self._still_yaw_speed
            and joints < self._still_joint_speed
            and spread < self._still_feet_spread
        )

    def _stand_in_place(self) -> torch.Tensor:
        """Stand stiff with the legs where they are [rad]: no jump to the default posture."""
        self.reset()
        self._stand_pose = self.robot.data.joint_pos.torch[:, self._leg_joint_ids].clone()
        return self._stand_pose.clone()

    def _compute_observation(self, command: torch.Tensor) -> torch.Tensor:
        terms = [
            self.robot.data.root_lin_vel_b.torch[:, :3],
            self.robot.data.root_ang_vel_b.torch[:, :3],
            self.robot.data.projected_gravity_b.torch[:, :3],
            command,
            self.robot.data.joint_pos.torch[:, self._leg_joint_ids] - self._default_leg_pos,
            self.robot.data.joint_vel.torch[:, self._leg_joint_ids],
            self._last_action,
        ]
        if self._arm_joint_ids is not None:
            terms += [
                self.robot.data.joint_pos.torch[:, self._arm_joint_ids] - self._arm_default,
                self.robot.data.joint_vel.torch[:, self._arm_joint_ids],
            ]
        return torch.cat(terms, dim=-1).to(dtype=torch.float32)
