# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fakes for the unit tests, built through real constructors where possible.

Joint worlds for :class:`EpisodeRuntime`, a scene for a real ``IsaacLabWorld``, observation and
navigation stand-ins and a ROS-free Nav2 transport. The UI fakes are in
``ui_fakes``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from phygs.execution.locomotion.base import DeadBand
from phygs.execution.manipulation.controller import ManipulationController, MotionConfig
from phygs.execution.navigation.fine_approach import FineApproach
from phygs.execution.navigation.map import GridMap
from phygs.robots.description import load_robot
from phygs.runtime.episode import EpisodeRuntime, EpisodeStatus, JointTargets, RunMode
from phygs.sim.observations import IsaacLabObservations, pose_matrix
from phygs.skills.manipulation.api import ManipulationClient

# Spot, the robot the fakes stand in for: its description, the transit posture navigation checks, and
# its base's command dead band and lowest-speed moves (navigation's fine approach).
SPOT = load_robot("spot")
TRANSIT = SPOT.arm.transit_posture
SPOT_DEAD_BAND = DeadBand.load(SPOT.base.config)
SPOT_COVERAGE = FineApproach.Coverage.from_moves(yaml.safe_load(SPOT.base.config.read_text())["lowest_speed_moves"])


def robot_parts(arm_joints, gripper, open_angle, closed_angle, tools=None):
    """The parts of a robot description a manipulation client reads: the arm joints, a one-joint
    gripper (open and closed positions [rad]) and the tool frames (default: both at the wrist)."""
    tools = tools or {"wrist": np.eye(4), "hand": np.eye(4)}
    gripper = SimpleNamespace(joints=(gripper,), primary=gripper, open=open_angle, closed=closed_angle, tools=tools)
    return SimpleNamespace(arm=SimpleNamespace(joints=tuple(arm_joints)), gripper=gripper)


# Joint worlds ------------------------------------------------------------------------------------


class FakeWorld:
    """A world whose joints follow their position targets [rad] one physics tick after they are set.

    Args:
        joint_names: The joints, in order.
        initial: The joint positions [rad] restored at each reset: one value for every joint, or
            values by joint (the others start at 0).
        physics_dt: The physics step [s]; the project runs at 250 Hz.
        follow: Per-tick behaviour: maps the targets to the positions reached this tick
            (default: the targets).

    It records ``steps`` (physics ticks since the last restore), ``target`` (the applied position
    targets) and ``gains`` (stiffness [N·m/rad] by joint, explicit or PhysX drive, from
    :class:`JointTargets`). Velocities [rad/s] are the last tick's finite difference.
    """

    def __init__(
        self,
        joint_names: tuple[str, ...],
        *,
        initial: float | dict[str, float] = 0.0,
        physics_dt: float = 0.004,
        follow: Callable[[dict[str, float]], dict[str, float]] | None = None,
    ):
        self.physics_dt = physics_dt
        self.joint_names = tuple(joint_names)
        self.initial = initial
        if follow is not None:
            self.follow = follow
        self.restore()

    def restore(self):
        if isinstance(self.initial, dict):
            self.state = {**dict.fromkeys(self.joint_names, 0.0), **self.initial}
        else:
            self.state = dict.fromkeys(self.joint_names, self.initial)
        self.target = self.state.copy()
        self.velocity = dict.fromkeys(self.joint_names, 0.0)
        self.gains = {}
        self.steps = 0

    def positions(self):
        return self.state.copy()

    def velocities(self):
        return self.velocity.copy()

    def apply(self, targets):
        if isinstance(targets, JointTargets):
            self.gains.update(targets.stiffness)
            self.gains.update(targets.drive_stiffness)
            targets = targets.positions
        self.target.update(targets)

    def follow(self, target):
        """The positions [rad] the joints reach this tick: the targets."""
        return dict(target)

    def step(self):
        new = self.follow(self.target)
        self.velocity = {j: (new[j] - self.state[j]) / self.physics_dt for j in self.joint_names}
        self.state = new
        self.steps += 1


class ManipulationWorld(FakeWorld):
    """An arm and a claw [rad] at 100 Hz, with the faults of a blocked, loaded or squeezing arm.

    Args:
        stuck: No joint moves.
        object_at: The claw angle [rad] where an object blocks closing.
        contact_velocity: The claw speed [rad/s] reported while blocked, as PhysX does in contact.
        chatter: A blocked claw vibrates by +-chatter [rad] each tick.
        sag: The arm settles this far [rad] below its target, as under a held load.

    ``creep`` [rad per tick]: a squeezed object yields this far each tick, as a soft or slipping one does.
    """

    def __init__(self, *, stuck=False, object_at=None, contact_velocity=0.0, chatter=0.0, sag=0.0):
        self.stuck, self.object_at = stuck, object_at
        self.sag = sag
        self.contact_velocity = contact_velocity
        self.chatter, self.flip = chatter, 1.0
        self.creep = 0.0
        super().__init__(("arm", "claw"), physics_dt=0.01)

    def velocities(self):
        velocity = super().velocities()
        if self.object_at is not None and self.state["claw"] == self.object_at:
            velocity["claw"] = self.contact_velocity
        return velocity

    def follow(self, target):
        if self.creep and self.object_at is not None and target["claw"] > self.object_at:
            self.object_at = min(self.object_at + self.creep, 0.0)
        new = dict(self.state if self.stuck else target)
        if self.object_at is not None and self.state["claw"] <= self.object_at + self.chatter:
            new["claw"] = min(new["claw"], self.object_at)
            if new["claw"] == self.object_at and self.chatter:
                new["claw"] += self.chatter * self.flip
                self.flip = -self.flip
        if not self.stuck:
            new["arm"] = target["arm"] - self.sag
        return new


def manipulation_stack(
    world,
    limits,
    *,
    gripper,
    open_angle,
    closed_angle,
    enabled=True,
    source=RunMode.AUTONOMOUS,
    kinematics=None,
    tools=None,
):
    """A real controller, runtime and client over ``world``, reset for the first episode.

    Args:
        world: The joint world.
        limits: Joint limits [rad] by joint.
        gripper: The gripper joint.
        open_angle: The gripper's open angle [rad].
        closed_angle: The gripper's closed angle [rad].
        enabled: Whether the task permits manipulation.
        source: The controller's command source.
        kinematics: ``(kinematics, observations)`` bound to the client, if any.
        tools: The tool frames in the wrist (4x4) [m] by name (default: ``wrist`` and ``hand`` at the wrist).

    Returns:
        ``(world, controller, runtime, client)``.
    """
    controller = ManipulationController(
        world.joint_names, world, limits, MotionConfig(), source=source, gripper=gripper
    )
    runtime = EpisodeRuntime(world, (controller,), mode=RunMode.AUTONOMOUS)
    arm = tuple(joint for joint in world.joint_names if joint != gripper)
    client = ManipulationClient(
        runtime, controller, robot_parts(arm, gripper, open_angle, closed_angle, tools), enabled=enabled
    )
    if kinematics is not None:
        client.bind_kinematics(*kinematics)
    runtime.reset()
    return world, controller, runtime, client


def setup(
    *,
    stuck=False,
    enabled=True,
    source=RunMode.AUTONOMOUS,
    object_at=None,
    contact_velocity=0.0,
    chatter=0.0,
    sag=0.0,
    limits=None,
    tools=None,
):
    """The manipulation stack over a :class:`ManipulationWorld`: claw open at -1 rad, closed at 0 rad;
    ``tools`` as :func:`manipulation_stack`'s.

    Returns:
        ``(world, controller, runtime, client)``.
    """
    world = ManipulationWorld(
        stuck=stuck, object_at=object_at, contact_velocity=contact_velocity, chatter=chatter, sag=sag
    )
    limits = limits or dict.fromkeys(world.joint_names, (-2, 2))
    return manipulation_stack(
        world, limits, gripper="claw", open_angle=-1, closed_angle=0, enabled=enabled, source=source, tools=tools
    )


def advance(runtime, client, seconds):
    """Step the runtime for ``seconds`` [s], updating the client after each tick."""
    for _ in range(round(seconds / runtime.physics_dt)):
        runtime.step()
        client.update()


# IsaacLab scene ----------------------------------------------------------------------------------


class FakeRobot:
    """An IsaacLab articulation: joint state and the targets written to it.

    Positions [rad] and velocities [rad/s] are one value for every joint. Position-target writes
    are kept in ``writes`` as ``(joint_ids, target)``; the last velocity and effort targets in
    ``velocity_target`` and ``effort_target``.
    """

    device = "cpu"
    num_base_dofs = 6  # A floating base precedes the joints in the generalized coordinates.

    def __init__(self, joint_names: tuple[str, ...], *, position: float = 0.0, velocity: float = 0.0):
        count = len(joint_names)
        self.joint_names = tuple(joint_names)
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.full((1, count), position)),
            joint_vel=SimpleNamespace(torch=torch.full((1, count), velocity)),
        )
        self.actuators = {}
        self.writes = []
        self.velocity_target = self.effort_target = None

    def set_joint_position_target_index(self, *, target, joint_ids=None):
        self.writes.append((joint_ids, target.clone()))

    def set_joint_effort_target_index(self, *, target, joint_ids=None):
        self.effort_target = target.clone()

    def set_joint_velocity_target_index(self, *, target, joint_ids=None):
        self.velocity_target = target.clone()


class FakeScene(dict):
    """An IsaacLab interactive scene for :func:`isaaclab_world`: one environment, entities by name.

    Args:
        robot: The articulation ``robot``.
        sensors: Sensors by name.
        collections: Rigid-object collections by name (the scene bodies).

    It records ``restored`` (the states passed to ``reset_to``), ``writes`` (``write_data_to_sim``
    calls) and ``ticks`` (physics updates since the last reset). A test that needs another reaction
    replaces the method on the instance.
    """

    num_envs = 1

    def __init__(self, robot, *, sensors=None, collections=None):
        sensors, collections = dict(sensors or {}), dict(collections or {})
        super().__init__(robot=robot, **sensors, **collections)
        self.articulations = {"robot": robot}
        self.sensors, self.rigid_object_collections = sensors, collections
        self.deformable_objects, self.surface_grippers = {}, {}
        self.restored, self.writes, self.ticks = [], 0, 0

    def get_state(self):
        return {}

    def reset(self):
        self.ticks = 0

    def reset_to(self, state):
        self.restored.append(state)

    def write_data_to_sim(self):
        self.writes += 1

    def update(self, dt):
        self.ticks += int(dt > 0)


def isaaclab_world(scene, **options):
    """A real ``IsaacLabWorld`` over ``scene``, stepping a headless 250 Hz simulation.

    Args:
        scene: The scene, e.g. a :class:`FakeScene`.
        options: Keyword arguments of the ``IsaacLabWorld`` constructor.
    """
    from phygs.sim.state import IsaacLabWorld

    sim = SimpleNamespace(get_physics_dt=lambda: 0.004, is_rendering=False, step=lambda **kwargs: None)
    return IsaacLabWorld(sim, scene, **options)


# Observations ------------------------------------------------------------------------------------


def array(value, dtype=torch.float32):
    """A tensor under ``.torch``, as IsaacLab presents asset and sensor data."""
    return SimpleNamespace(torch=torch.tensor(value, dtype=dtype))


class ObservationScene(dict):
    """A robot at (4, 5, 6) m, a 1x2-pixel ``front`` RGB-D camera 0.5 m ahead, and an IMU."""

    num_envs = 1
    device = "cpu"

    def __init__(self):
        robot = SimpleNamespace(
            joint_names=["leg", "arm"],
            data=SimpleNamespace(
                joint_pos=array([[0.1, 0.2]]),
                joint_vel=array([[0.0, 0.0]]),
                root_pose_w=array([[4, 5, 6, 0, 0, 0, 1]]),
                root_lin_vel_b=array([[0.3, 0, 0]]),
            ),
        )
        camera = SimpleNamespace(
            cfg=SimpleNamespace(
                data_types=["rgb", "distance_to_image_plane"], update_latest_camera_pose=True, update_period=0.0
            ),
            data=SimpleNamespace(
                pos_w=array([[4.5, 5, 6]]),
                quat_w_ros=array([[0, 0, 0, 1]]),
                intrinsic_matrices=array([[[2, 0, 0], [0, 2, 0], [0, 0, 1]]]),
                output={
                    "rgb": array([[[[255, 0, 0], [0, 255, 0]]]], torch.uint8),
                    "distance_to_image_plane": array([[[[2], [float("inf")]]]]),
                },
            ),
        )
        imu = SimpleNamespace(data=SimpleNamespace(ang_vel_b=array([[0, 0, 0.1]]), lin_acc_b=array([[0, 0, 9.81]])))
        super().__init__(robot=robot, front=camera, observation_imu=imu)
        self.sensors = {"front": camera, "observation_imu": imu}


class FakeCameras:
    """Camera source over the fake scene's camera data; ``render`` runs before each capture. Depth is
    delivered on ``depth_device``, as the renderer delivers it on the GPU."""

    def __init__(self, scene, render=None, depth_device="cpu"):
        self._scene, self._render, self._depth_device = scene, render, depth_device
        self.names = tuple(name for name in scene.sensors if name != "observation_imu")

    def data_types(self, name):
        return tuple(self._scene[name].cfg.data_types)

    def render(self):
        if self._render is not None:
            self._render()

    def read(self, name):
        data = self._scene[name].data
        output = data.output
        return SimpleNamespace(
            rgb=output["rgb"].torch[0, ..., :3].clone() if "rgb" in output else None,
            depth=output["distance_to_image_plane"].torch[0, ..., 0].to(self._depth_device, copy=True),
            intrinsics=data.intrinsic_matrices.torch[0].clone(),
            world_from_camera=pose_matrix(data.pos_w.torch[0], data.quat_w_ros.torch[0]),
        )


def observations(scene, config, dt=0.004, **kwargs):
    """The adapter with the fake scene's cameras, at the project's 250 Hz physics step [s]."""
    cameras = FakeCameras(scene, kwargs.pop("render", None), kwargs.pop("depth_device", "cpu"))
    return IsaacLabObservations(scene, config, dt, base=RootBase(scene["robot"]), cameras=cameras, **kwargs)


class RootBase:
    """The base link of a fake robot whose root is the base: its root pose and velocity."""

    def __init__(self, robot):
        self._robot = robot

    def pose(self):
        return self._robot.data.root_pose_w.torch[0]

    def linear_velocity_b(self):
        return self._robot.data.root_lin_vel_b.torch[0]


# Navigation --------------------------------------------------------------------------------------


def grid(occupied=None, version=0):
    """A free 8 m x 8 m map of 0.2 m cells from (-2, -4) m, or ``occupied`` cells on it."""
    return GridMap(
        np.zeros((40, 40), dtype=bool) if occupied is None else occupied, (-2.0, -4.0), 0.2, version, "test_known_map"
    )


class Runtime:
    """An episode runtime ticking every 0.1 s, localized at a settable planar pose [m, rad]."""

    physics_dt = 0.1
    episode_id = 1
    status = EpisodeStatus.RUNNING
    tick = 0

    def __init__(self):
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.localized = True
        self.velocity = torch.zeros(3)
        self.joints = dict(TRANSIT)

    @property
    def time_seconds(self):
        return self.tick * self.physics_dt

    def observe(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pose = torch.tensor([[c, -s, 0, self.x], [s, c, 0, self.y], [0, 0, 1, 0.5], [0, 0, 0, 1.0]])
        localization = (
            SimpleNamespace(
                mode="ground_truth",
                linear_velocity=self.velocity,
                valid=True,
                pose=SimpleNamespace(matrix=pose, target="world"),
                stamp=SimpleNamespace(time_seconds=self.time_seconds),
            )
            if self.localized
            else None
        )
        return SimpleNamespace(
            localization=localization,
            joints=SimpleNamespace(
                valid=True, names=tuple(self.joints), positions=torch.tensor(list(self.joints.values()))
            ),
        )


class Motion:
    """Locomotion commands on a :class:`Runtime`: expiry starts an 8-tick (0.8 s) controlled stop."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.records = {}
        self.command = (0.0, 0.0, 0.0)
        self.sent = []  # (command, duration [s]) in order

    def command_velocity(self, x, y, yaw, *, duration):
        for record in self.records.values():
            if record.status in {"active", "stopping"}:
                record.status = "overridden"
        self.command = (x, y, yaw)
        command_id = len(self.records) + 1
        self.records[command_id] = SimpleNamespace(status="active", expiry=self.runtime.tick + round(duration / 0.1))
        self.sent.append((self.command, duration))
        return command_id

    def stop(self):
        command_id = self.command_velocity(0, 0, 0, duration=0.8)
        self.records[command_id].status = "stopping"
        self.stop_deadline = self.runtime.tick + 8
        return command_id

    _command_velocity = command_velocity
    _stop = stop

    def feedback(self, command_id):
        record = self.records[command_id]
        if record.status == "active" and self.runtime.tick >= record.expiry:  # As the runtime: expiry stops.
            record.status, self.command, self.stop_deadline = "stopping", (0.0, 0.0, 0.0), self.runtime.tick + 8
        if record.status == "stopping" and self.runtime.tick >= self.stop_deadline:
            record.status = "done"
        return record


# ROS ---------------------------------------------------------------------------------------------


def nav2_transport(config=None, **attributes):
    """A ``RclpyNav2Transport`` without ROS, for its bookkeeping and message handling.

    The real constructor starts a ROS context and node, so this bypasses it: it sets the state the
    constructor sets, ``Mock`` ROS handles, then ``attributes`` as given (the transport's own
    names, private ones included), which may also replace methods.

    Args:
        config: The transport configuration (default: ``Nav2Config()``).
        attributes: Attributes to set or replace.
    """
    from unittest.mock import Mock

    from phygs.execution.navigation.backends.nav2 import Nav2Config
    from phygs.execution.navigation.backends.ros2 import RclpyNav2Transport

    config = Nav2Config() if config is None else config
    epoch = 1_000_000_000
    state = dict(
        _config=config,
        _geometry={},
        _poisoned=False,
        _raw_map=None,
        _acknowledged_map=-1,
        _clear_pending=True,
        _epoch=epoch,
        _stamp=epoch,
        _minimum_stamp=epoch,
        _map_version=-1,
        _camera_stamps={},
        _last_cameras={},
        _command=None,
        _planning=None,
        _following=None,
        _plan_result=None,
        _follow_result=None,
        _state=None,
        _closed=False,
        _explorer=None,
        _navigate_server=None,
        _explore_status=None,
        _explore_goals=[],
        _explore_current=None,
        _explore_outcomes={},
        _explore_complete=False,
        _latest_costmap=None,
    )
    handles = ("_rclpy", "_context", "_node", "_executor", "_clock", "_odom", "_map", "_path", "_tf")
    state.update({name: Mock() for name in handles})
    space = "/" + config.namespace  # One Nav2 stack on the domain.
    state["_node"].get_node_names_and_namespaces.return_value = [
        ("planner_server", space),
        ("controller_server", space),
    ]
    state.update({name: Mock() for name in ("_subscription", "_planner", "_controller", "_parameters")})
    state["_points"] = {name: Mock() for name in config.sensor_cameras}
    state["_clearing"] = {name: Mock() for name in config.sensor_cameras}
    for name in ("_costmap_parameters", "_costmaps", "_clear_costmaps", "_lifecycle", "_transitions"):
        state[name] = [Mock(), Mock()]
    state.update(attributes)
    transport = object.__new__(RclpyNav2Transport)
    for name, value in state.items():
        setattr(transport, name, value)
    return transport
