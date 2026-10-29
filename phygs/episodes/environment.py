# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Environment: builds an episode from its spec, runs an agent, and writes the report.

Needs a running Isaac Sim app (``AppLauncher`` before import). The Environment owns the
simulation loop on the main thread; the agent runs ``run(session)`` in its own thread
(:mod:`.session`). It also holds what the agent must not: the scene's ground truth
(:meth:`Environment.truth`, for robot tests), the task (its evaluator) and the recorder. One
Environment per process.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
import yaml
from isaaclab.scene import InteractiveScene

from phygs.episodes.hints import PrivilegedHints
from phygs.episodes.provenance import code_version, file_sha256, usd_sha256
from phygs.episodes.recorder import Recorder
from phygs.episodes.report import REPORT_SCHEMA
from phygs.episodes.session import (
    AgentFailed,
    AgentSession,
    ArticulatedPart,
    Audit,
    EpisodeEnded,
    MainThread,
    ObjectTarget,
    PublicTask,
)
from phygs.episodes.spec import SCHEMA, EpisodeSpec, load_agent
from phygs.episodes.task import NoTask, Task
from phygs.episodes.task import summary as task_summary
from phygs.episodes.viewer import EpisodeView, Viewer
from phygs.execution.manipulation.arm_planner import ArmPlanner
from phygs.execution.manipulation.kinematics import ArmKinematics
from phygs.execution.manipulation.planners.model import collider_spheres, usd_only_boxes
from phygs.execution.manipulation.skills.backends import STOW_TOLERANCE_RAD, create_skills
from phygs.execution.manipulation.skills.types import SkillConfig
from phygs.execution.navigation.backends.nav2_stack import Nav2Stack
from phygs.execution.navigation.footprint import transit_footprint
from phygs.observations.types import CameraPermission, ObservationAccess, ObservationConfig
from phygs.robots.control import RobotControl
from phygs.robots.description import RobotDescription, load_robot
from phygs.robots.scene import make_scene_cfg, robot_cameras, simulation_cfg
from phygs.runtime.episode import EpisodeStatus, RunMode
from phygs.runtime.pipeline import PipelineConfig, ServiceDeclaration
from phygs.simulation.base import BaseLink
from phygs.simulation.maps import BoxGeometry
from phygs.simulation.navigation import nav2_navigation
from phygs.simulation.scene import UsdScene, pose_matrix
from phygs.tools.panels import PanelFeed

PROJECT = Path(__file__).resolve().parents[2]  # The PhyGS checkout.
SETTINGS = PROJECT / "phygs" / "configs" / "environment.yaml"
ROBOT_PATH = "/World/envs/env_0/Robot"
CLOCKS = ("charged", "realtime")
# Grants a client uses without a call naming them (the report lists them as implicit).
IMPLICIT = {
    "gt_localization": "navigation and skills localize with ground truth",
    "known_map": "navigation and the arm planner avoid the scene's known boxes (without it, Nav2 maps the"
    " navigation cameras' depth and the arm plans on depth)",
}


class Truth:
    """Ground truth now, for the evaluator and robot tests; never given to agents."""

    def __init__(self, environment: Environment) -> None:
        self._scene = environment.scene_def
        self._robot = environment.scene["robot"]
        self._description = environment.robot
        self._base = environment.base

    def object_pose(self, name: str) -> np.ndarray:
        """World pose (4x4) [m] of a scene object."""
        return self._scene.object_pose(name)

    def part(self, name: str) -> ArticulatedPart:
        """An articulated part now (its ``opening`` [m])."""
        return self._scene.part(name)

    def held(self, name: str, *, lift_m: float, reach_m: float) -> bool:
        """Whether an object is in the hand: lifted at least ``lift_m`` [m] above where it started
        and its centre within ``reach_m`` [m] of the wrist link (grasped objects are never attached,
        so an object off its support next to the wrist is held). The thresholds are the evaluator's."""
        center = self._scene.object_box(name)[0]
        start = self._scene.initial_box(name)[0]
        wrist_link = self._robot.body_names.index(self._description.arm.wrist_link)
        wrist = self._robot.data.body_link_pose_w.torch[0, wrist_link, :3]
        near = float(np.linalg.norm(center - wrist.cpu().numpy())) <= reach_m
        return bool(center[2] - start[2] >= lift_m and near)

    def part_openings(self) -> dict[str, float]:
        """Every drawer's opening [m] now."""
        return {name: self._scene.part(name).opening for name in self._scene.part_names}

    def inside(self, name: str, part: str) -> bool:
        """Whether an object's centre is inside a part (a drawer)."""
        return self._scene.inside(self._scene.object_box(name)[0], part)

    def base_pose(self) -> np.ndarray:
        """World pose (4x4) [m] of the robot's base."""
        return pose_matrix(self._base.pose().cpu().tolist())

    def arm_joints(self) -> dict[str, float]:
        """Arm joint positions [m or rad], planning order."""
        positions = self._robot.data.joint_pos.torch[0].cpu().tolist()
        return {name: positions[self._robot.joint_names.index(name)] for name in self._description.arm.joints}

    def stow_error_rad(self) -> float:
        """Largest arm joint distance [m or rad, rad for a revolute arm] from the stow posture."""
        stow = self._description.arm.stow
        return float(max(abs(value - goal) for value, goal in zip(self.arm_joints().values(), stow, strict=True)))


class Environment:
    """One episode: ``Environment(spec, output)``, then :meth:`run` an agent; close after use.

    A ``viewer`` (:class:`~phygs.episodes.viewer.Viewer`, e.g. the robot UI's page)
    watches alongside the agent: its cameras are captured through the declared ``viewer``
    observation consumer, and it gets a read-only :class:`~phygs.episodes.viewer.EpisodeView`
    on the main thread once per camera period. Its reads are not agent calls. With
    ``hold_clock``, the realtime clock starts only at the agent's first call (a person reads the
    task before pressing Start); the viewer keeps updating meanwhile.

    The robot is the spec's (``robot.model``); ``robot`` (a description, a robot name or the path of
    a ``robot.yaml``) runs the episode with another one.

    ``overrides`` is a mapping applied over the settings file (top-level keys replace; a mapping
    under a key merges one level, e.g. ``{"planner": {"backend": "moveit"}}``), so a benchmark
    can pin its end-of-episode and backend settings without writing a copy of the file.

    ``task`` is the task object built from the spec's task block
    (:func:`phygs_robot_tests.tasks.load_task`): the agent's brief, the privileged target and the
    evaluation. The Environment never reads the block itself; without a task object a spec whose
    block is not ``{goal: none}`` is refused.
    """

    def __init__(
        self,
        spec: EpisodeSpec,
        output: Path,
        *,
        clock: str = "charged",
        record: bool = False,
        device: str = "cpu",
        settings_path: Path = SETTINGS,
        viewer: Viewer | None = None,
        hold_clock: bool = False,
        robot: RobotDescription | str | Path | None = None,
        task: Task | None = None,
        overrides: dict | None = None,
    ) -> None:
        if clock not in CLOCKS:
            raise ValueError(f"Unknown clock {clock!r}; expected one of {CLOCKS}.")
        if task is None:
            if spec.task != {"goal": "none"}:
                raise ValueError("The spec has a task block; pass its task object (phygs_robot_tests.tasks.load_task).")
            task = NoTask()
        self.task = task
        output = Path(output)
        if (output / "report.json").exists():
            raise FileExistsError(f"{output} holds a finished episode; choose a fresh output folder.")
        output.mkdir(parents=True, exist_ok=True)
        self.spec, self.output, self.clock, self._record = spec, output, clock, record
        self.viewer, self._hold_clock = viewer, hold_clock
        self._last_view = -math.inf  # Wall time [s] of the viewer's last update while the clock is held.
        settings = yaml.safe_load(settings_path.read_text())
        for key, value in (overrides or {}).items():
            if isinstance(value, dict) and isinstance(settings.get(key), dict):
                settings[key] = {**settings[key], **value}
            else:
                settings[key] = value
        self.settings = settings
        if not isinstance(robot, RobotDescription):
            robot = load_robot(spec.robot.model if robot is None else robot)
        # The spec's start for this robot (robot.starts), and the cameras its roles name on it.
        self.spec = spec = spec.with_robot(robot.name)
        self.robot = robot
        self.observed_cameras = spec.observations.for_robot(robot)
        self._check_supported(spec, settings, robot)
        manipulation = "manipulation" in spec.actions
        self.skill_config = SkillConfig.from_robot(robot, settings["skills"]["backend"]) if manipulation else None
        self.scene_def = UsdScene(
            spec.scene.name,
            spec.scene.usd,
            ground=spec.scene.ground,
            objects=spec.objects,
            navigation_map=settings["map"],
            robot_start=spec.robot.start,
        )
        dt, period = settings["physics_dt"], settings["camera_period"]
        self.render_ticks = round(period / dt)
        self.scene_def.open_stage()  # A scene file, before the simulation context exists.
        self.sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, render_interval=self.render_ticks, device=device))
        x, y, heading = spec.robot.start
        view = self._record_view()
        cameras = robot_cameras(robot, self._captured_cameras())
        if record:
            eye, target = view
            cameras["third_person"] = dict(
                prim_path="{ENV_REGEX_NS}/ThirdPerson",
                **settings["record"],
                data_types=["rgb"],
                offset=dict(pos=eye.tolist(), rot=_look_rotation(eye, target), convention="world"),
                pinhole=dict(focal_length=20.0, horizontal_aperture=20.955, clipping_range=[0.05, 30.0]),
            )
        cfg = make_scene_cfg(
            robot,
            physics_dt=dt,
            light=settings["light"],
            position=(x, y, robot.spawn_height),
            heading=heading,
            cameras=cameras,
        )(num_envs=1, env_spacing=3.0, replicate_physics=False)
        self.scene_def.define(self.sim.stage, cfg, dt, contact_links=robot.arm_links)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.scene_def.bind(self.scene)
        robot_asset = self.scene["robot"]
        self.base = BaseLink(robot_asset, robot.base_link)
        self.scene.update(dt)
        self.sim.set_camera_view(*(tuple(v.tolist()) for v in view))
        self._nav2: Nav2Stack | None = None  # Nav2's servers, when navigation runs on Nav2.
        self.control = self._create_control(spec, settings, record)
        self.planner = None
        if "manipulation" in spec.actions:
            self.planner = ArmPlanner(
                robot,
                project=PROJECT,
                output=output,
                settings=settings["planner"],
                geometry=collider_spheres(
                    self.sim.stage, ROBOT_PATH, robot.planning, cell_size=settings["planner"]["geometry_cell_size"]
                ),
                usd_boxes=usd_only_boxes(self.sim.stage, ROBOT_PATH, robot.planning),
                observe=self.control.pipeline.view("skills").observe,
                # Without known_map the arm plans on the robot's own depth cameras.
                cameras=() if self._known_map else robot.cameras_for("planning"),
                boxes=self._obstacles if self._known_map else None,
                target=task.target,
                stow=self.skill_config.planner_stow(STOW_TOLERANCE_RAD),
                episode_id=lambda: self.control.runtime.episode_id,
                time_seconds=lambda: self.control.runtime.time_seconds,
            )
        self._panels = None  # The video's panels beside the third-person view (helpers.panels).
        if record:
            outline = None  # The transit outline the navigation panel draws, once it is measured.
            if robot.navigation.footprint_profile is not None:
                profile = yaml.safe_load(robot.navigation.footprint_profile.read_text())
                x0, x1, y0, y1 = transit_footprint(profile).bounds()
                outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            self._panels = PanelFeed(self.control, footprint=outline, planner=self.planner)
        self.provenance.update(task.provenance())
        self.recorder = Recorder(
            output,
            video=record,
            fps=1.0 / period,
            jaw_links=robot.gripper.jaw_links,
            read_ticks=settings["contact_read_ticks"],
        )
        self.report: dict | None = None
        self._during: Callable[[Environment], None] | None = None  # The robot test's per-tick call (run).
        self._reset_done = False
        self._finished: str | None = None
        self._outcome: tuple[str, str] | None = None
        self._last_render = 0.0

    def _record_view(self) -> tuple[np.ndarray, np.ndarray]:
        """The third-person camera (eye, target) [m]: the spec's, else behind and above the robot's
        start, looking ahead of it."""
        if self.spec.record_view is not None:
            return tuple(np.array(self.spec.record_view[key], dtype=float) for key in ("eye", "target"))
        x, y, heading = self.spec.robot.start
        c, s = math.cos(heading), math.sin(heading)
        eye = np.array([x - 2.5 * c + 1.5 * s, y - 2.5 * s - 1.5 * c, 2.3])
        return eye, np.array([x + 1.0 * c, y + 1.0 * s, 0.4])

    @staticmethod
    def _check_supported(spec: EpisodeSpec, settings: dict, robot: RobotDescription) -> None:
        """Fail fast on combinations the harness cannot build yet (no silent substitutes)."""
        unknown = set(spec.observations.cameras) - set(robot.cameras)
        unknown |= {role for role in spec.observations.camera_roles if not robot.cameras_for(role)}
        if unknown:
            raise ValueError(f"Unknown cameras {sorted(unknown)}; {robot.name} has {sorted(robot.cameras)}.")
        grants = set(spec.grants)
        if "navigation" in spec.actions:
            if "gt_localization" not in grants:
                raise NotImplementedError(
                    "Navigation without the gt_localization grant needs a localization estimator, not wired into"
                    " episodes yet."
                )
            if "known_map" not in grants and not robot.cameras_for("navigation"):
                raise ValueError(f"Navigation without known_map maps depth, and {robot.name} has no navigation camera.")
        if "manipulation" in spec.actions:
            if "gt_localization" not in grants:
                raise NotImplementedError(
                    "Manipulation without the gt_localization grant needs a localization estimator, not wired into"
                    " episodes yet."
                )
            if robot.look_camera is None or ("known_map" not in grants and not robot.cameras_for("planning")):
                raise ValueError(
                    f"Manipulation needs a look camera and, without known_map, planning cameras; {robot.name}"
                    " lacks them."
                )

    @property
    def _known_map(self) -> bool:
        return "known_map" in self.spec.grants

    def _captured_cameras(self) -> tuple[str, ...]:
        """The agent's cameras, plus the skills' look camera and, without known_map, the arm planner's
        and navigation's; when recording, also the video panels' (the robot's front and look cameras);
        with a viewer, also the viewer's."""
        robot = self.robot
        cameras = list(self.observed_cameras)
        if self.viewer is not None:
            unknown = set(self.viewer.cameras) - set(robot.cameras)
            if unknown:
                raise ValueError(f"The viewer asks for unknown cameras {sorted(unknown)}")
            cameras += [name for name in self.viewer.cameras if name not in cameras]
        if self._record:
            cameras += [name for name in PanelFeed.cameras(robot) if name not in cameras]
        if "manipulation" in self.spec.actions:
            planner = () if self._known_map else robot.cameras_for("planning")
            cameras += [name for name in (robot.look_camera, *planner) if name not in cameras]
        if "navigation" in self.spec.actions and not self._known_map:
            cameras += [name for name in robot.cameras_for("navigation") if name not in cameras]
        return tuple(cameras)

    def _create_control(self, spec, settings, record) -> RobotControl:
        dt, period = settings["physics_dt"], settings["camera_period"]
        observed = tuple(CameraPermission(name) for name in self.observed_cameras)
        captured = tuple(CameraPermission(name) for name in self._captured_cameras())
        recording = (CameraPermission("third_person", depth=False),) if record else ()
        capture = ObservationConfig(
            joint_state=True,
            imu=spec.observations.imu,
            localization="ground_truth",
            state_period=dt,
            camera_period=period,
            cameras=captured + recording,
        )
        agent = ObservationAccess(
            joint_state=spec.observations.joint_state,
            imu=spec.observations.imu,
            localization="ground_truth" if "gt_localization" in spec.grants else "none",
            cameras=observed,
        )
        consumers = [
            # The skills read joints, the base pose and their camera, whatever the agent observes.
            ServiceDeclaration(
                "skills", ObservationAccess(joint_state=True, localization="ground_truth", cameras=captured)
            ),
        ]
        if self.viewer is not None:  # A person watching: the viewer's cameras, the joints and the base pose.
            shown = tuple(item for item in captured if item.name in self.viewer.cameras)
            access = ObservationAccess(joint_state=True, localization="ground_truth", cameras=shown)
            consumers.append(ServiceDeclaration("viewer", access))
        if record:  # The video and its panels: the cameras it shows and the base pose (helpers.panels).
            shown = tuple(item for item in captured if item.name in PanelFeed.cameras(self.robot))
            access = ObservationAccess(localization="ground_truth", cameras=recording + shown)
            consumers.append(ServiceDeclaration("recorder", access))
        self.provenance = dict(
            spec_schema=SCHEMA,
            robot=self.robot.summary(),
            robot_usd=usd_sha256(self.robot.usd),
            scene_usd=usd_sha256(spec.scene.usd),
            code=code_version(PROJECT),
        )
        navigation = {}
        privileged_sources = ()
        if "navigation" in spec.actions:
            navigation = self._navigation(settings, dt)
            privileged_sources = ("privileged_known_box_geometry",) if self._known_map else ()
        return RobotControl.create(
            self.sim,
            self.scene,
            self.robot,
            mode=RunMode.AUTONOMOUS,
            pipeline_config=PipelineConfig(capture, agent, ObservationAccess.from_config(capture), privileged_sources),
            observation_consumers=tuple(consumers),
            manipulation=True,
            manipulation_pose_planner=lambda pose, held=None: self.planner.request(world_pose=pose, held=held),
            manipulation_joint_planner=lambda goal, held=None: self.planner.request(goal=goal, held=held),
            allowed_actions=spec.actions,
            **navigation,
        )

    def _navigation(self, settings, dt) -> dict:
        """Nav2 with the measured transit footprint (its servers start here, in their own processes):
        on the scene's known boxes with known_map, else on the robot's navigation cameras' point
        clouds over the scene's map area, unknown beyond the start's ``explored_radius``."""
        boxes = None
        if self._known_map:
            boxes = tuple(
                BoxGeometry(tuple(map(float, size)), tuple(map(float, center)))
                for center, size in self.scene_def.map_boxes(exclude=(self.task.target,) if self.task.target else ())
            )
        navigation = nav2_navigation(
            stage=self.sim.stage,
            scene=self.scene,
            robot=self.robot,
            base=self.base,
            area=self.scene_def.map,
            dt=dt,
            output=self.output / "nav2",
            nav2=settings["navigation"]["nav2"],
            point_cloud=None if self._known_map else settings["navigation"]["point_cloud"],
            known_boxes=boxes,
            start=self.spec.robot.start[:2],
        )
        self._nav2 = navigation.stack
        # The files the base's motion depends on (a walking policy and its configuration) join the provenance.
        self.provenance.update({key: file_sha256(path) for key, path in navigation.base_files.items()})
        return navigation.control_arguments

    # Episode.
    def reset(self) -> None:
        """Seed, settle standing, start the arm planner with the settled legs and bind the skills."""
        torch.manual_seed(self.spec.seed)
        np.random.seed(self.spec.seed)
        self.control.reset(settling_seconds=self.settings["settling_seconds"])
        if self.planner is not None:
            self.planner.start()
            grants, robot = self.spec.grants, self.robot
            # Before the episode clock runs: the planner's first plan per gripper angle takes ~8 s.
            self.planner_warmup_seconds = self.planner.warm_up((robot.gripper.open,))
            kinematics = ArmKinematics.from_model(self.planner.model)
            self.control.manipulation.bind_kinematics(kinematics, self.control.pipeline.view("skills"))
            self.control.manipulation.bind_skills(
                create_skills(
                    self.control.manipulation,
                    self.control.pipeline.view("skills"),
                    self.planner.model,
                    self.skill_config,
                    ik=kinematics.inverse,
                    gripper_open=robot.gripper.positions(robot.gripper.open),
                    object_boxes=self.scene_def.object_box if "object_targets" in grants else None,
                    joint_planner=lambda goal, held=None: self.planner.request(goal=goal, held=held),
                    pose_planner=lambda pose, held=None: self.planner.request(world_pose=pose, held=held),
                )
            )
        self._openings_at_start = self.truth().part_openings()
        self._reset_done = True

    def run(
        self, agent: Callable | str, name: str | None = None, *, during: Callable[[Environment], None] | None = None
    ) -> dict:
        """Run ``agent(session)`` to its end, the budget, or an emergency stop; return and write the report.

        ``during`` is for robot tests, never agents: ``during(environment)`` runs on the main thread
        after every physics tick of the run, including the end settle, to sample ground truth or
        move a body mid-episode."""
        if self.report is not None:
            raise RuntimeError("This Environment ran its episode; build a new one.")
        if isinstance(agent, str):
            name, agent = load_agent(agent)
        name = name or getattr(agent, "__module__", "") + "." + getattr(agent, "__qualname__", repr(agent))
        if not self._reset_done:
            self.reset()
        runtime = self.control.runtime
        main = MainThread()
        audit = Audit(self.output / "audit.jsonl", lambda: runtime.time_seconds)
        clients = dict(
            navigation=self.control.navigation if "navigation" in self.spec.actions else None,
            manipulation=self.control.manipulation if "manipulation" in self.spec.actions else None,
            locomotion=self.control.locomotion if "motion" in self.spec.actions else None,
            arm=self.control.arm if "arm" in self.spec.actions else None,
        )
        session = AgentSession(
            task=PublicTask(self.task.goal, self.task.description),
            robot=self.robot.info(),
            grants=self.spec.grants,
            clients=clients,
            hints=PrivilegedHints(self.scene_def, self.task.target, self.spec.grants, boxes=self._obstacles),
            observe=self.control.observe,
            time_seconds=lambda: runtime.time_seconds,
            finish=self._finish,
            camera_period=self.settings["camera_period"],
            main=main,
            audit=audit,
            privileged=self._privileged,
        )
        thread = threading.Thread(target=self._agent_main, args=(agent, session), name="agent", daemon=True)
        start, wall = runtime.time_seconds, time.perf_counter()
        self._during = during
        thread.start()
        ended_by = reason = None
        broken = None  # An error in the Environment itself: reported, then raised.
        try:
            while ended_by is None:
                main.serve()
                if self._finished is not None:
                    ended_by, reason = "finish", self._finished
                elif not thread.is_alive():
                    ended_by, reason = self._outcome
                elif runtime.time_seconds - start >= self.spec.sim_seconds:
                    ended_by, reason = "budget", f"{self.spec.sim_seconds} simulated seconds"
                elif runtime.status != EpisodeStatus.RUNNING:
                    events = runtime.result()["events"]
                    ended_by, reason = f"runtime_{runtime.status.value}", events[-1]["detail"] if events else ""
                elif self.clock == "realtime" and self._hold_clock and main.served == 0 and not main.waiting:
                    main.serve(timeout=0.01)  # Time stands still until the agent's first call.
                    self._view_idle()
                    self._render_idle()
                elif self.clock == "realtime" or main.waiting:
                    self._tick()
                    main.wake()
                else:
                    main.serve(timeout=0.01)  # The agent thinks; time stands still.
                    self._render_idle()
        except Exception as error:
            broken = error
            ended_by, reason = "environment_error", traceback.format_exc()
        finally:
            main.end(ended_by or "the Environment stopped")
            thread.join(timeout=5.0)
        if self._outcome is None:  # Ended by the Environment while the agent was still running.
            self._outcome = ("agent_stopped", "")
        stopped = None
        if broken is None and runtime.status == EpisodeStatus.RUNNING:
            stopped = self._stop_all()
            for _ in range(round(self.settings["end_settle_seconds"] / runtime.physics_dt)):
                self._tick()
        self._during = None
        audit.close()
        contacts = self.recorder.contact_report()
        evaluation = (
            self.task.result(self.truth(), ended_by, contacts["contacts"], self._openings_at_start)
            if broken is None and runtime.status != EpisodeStatus.FAILED
            else {"outcome": "not_evaluated", "reason": f"the episode ended with {ended_by}"}
        )
        self.report = dict(
            schemas=dict(spec=SCHEMA, report=REPORT_SCHEMA),
            spec=self.spec.summary(),
            task=task_summary(self.task),
            provenance=self.provenance,
            agent=name,
            clock=self.clock,
            sim_seconds=round(runtime.time_seconds - start, 3),
            wall_seconds=round(time.perf_counter() - wall, 1),
            ended_by=ended_by,
            reason=reason,
            agent_outcome=dict(zip(("outcome", "detail"), self._outcome)),
            stopped=stopped,
            evaluation=evaluation,
            usage=audit.summary(),
            implicit_grants={grant: IMPLICIT[grant] for grant in self.spec.grants if grant in IMPLICIT},
            commands=audit.commands,
            **contacts,
            planner=None
            if self.planner is None
            else dict(
                backend=self.settings["planner"]["backend"],
                startup_seconds=self.planner.startup_seconds,
                warmup_seconds=getattr(self, "planner_warmup_seconds", None),
            ),
            narration=self.control.narration.log,
            runtime=self.control.runtime.result(),
        )
        self.recorder.write(self.report)
        if broken is not None:
            raise broken
        return self.report

    def _agent_main(self, agent: Callable, session: AgentSession) -> None:
        try:
            agent(session)
            self._outcome = ("agent_returned", "")
        except EpisodeEnded as error:
            self._outcome = ("agent_stopped", str(error))
        except AgentFailed as error:
            self._outcome = ("agent_failed", str(error))
        except BaseException:
            self._outcome = ("agent_error", traceback.format_exc())
            print(self._outcome[1], flush=True)

    def _finish(self, reason: str) -> None:
        self._finished = reason

    def _stop_all(self) -> dict:
        """Stop the robot after the agent ended, keeping the grip: cancel navigation and any arm
        command, stop walking, then tick until every controller is idle or the stop timeout passes.
        The gripper's hold setpoint survives (nothing calls ``clear_hold`` here), so a held target
        stays in the hand for the evaluation. Returns what was stopped and how long it took [s]."""
        control, runtime = self.control, self.control.runtime
        stopped = []
        navigation, manipulation, locomotion = control.navigation, control.manipulation, control.locomotion
        if navigation is not None and navigation.active_command is not None:
            navigation.cancel(navigation.active_command)
            stopped.append("navigation")
        if manipulation is not None and manipulation.active_command is not None:
            manipulation.cancel(manipulation.active_command)
            stopped.append("manipulation")
        if locomotion is not None and runtime.controller_state(locomotion.name) == "active" and not stopped:
            locomotion.stop()  # Walking under direct control; under navigation the cancel above stops it.
            stopped.append("locomotion")
        names = [client.name for client in (manipulation, locomotion) if client is not None]
        start = runtime.time_seconds
        timeout = self.settings["stop_all_timeout_seconds"]
        while runtime.status == EpisodeStatus.RUNNING and runtime.time_seconds - start < timeout:
            busy = (navigation is not None and navigation.active) or any(
                runtime.controller_state(name) != "idle" for name in names
            )
            if not busy:
                break
            self._tick()
        seconds = runtime.time_seconds - start
        return dict(commands=stopped, seconds=round(seconds, 3), timed_out=seconds >= timeout)

    def _obstacles(self, exclude: frozenset[str] = frozenset()) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The scene's colliders around the robot now (the arm planner's known boxes): within the
        robot's planning ``obstacle_radius`` of the base and in ``obstacle_height``, without the bodies
        in ``exclude``."""
        base = self.base.pose()[:2].cpu().numpy()
        height = tuple(self.settings["planner"]["obstacle_height"])
        return self.scene_def.obstacle_boxes(base, self.robot.planning.obstacle_radius, height, exclude)

    def _privileged(self, client: str, member: str, args: tuple, kwargs: dict) -> list[str]:
        """Grants a call reads through its arguments: an ObjectTarget names an object or a part."""
        used = []
        for value in (*args, *kwargs.values()):
            if isinstance(value, ObjectTarget):
                used.append("scene_hints" if value.object_id in self.scene_def.part_names else "object_targets")
        return used

    def _tick(self) -> None:
        """One physics tick with everything the Environment owns."""
        control, runtime = self.control, self.control.runtime
        control.step()
        if self.planner is not None:
            self.planner.poll()
        if runtime.tick % self.settings["contact_read_ticks"] == 0:
            # The running arm command labels its contacts; a skill's latest narrated step when it is
            # the one running (narration is not updated by arm commands, so it alone goes stale).
            active = control.timeline.active("arm") if control.timeline is not None else None
            latest = control.narration.latest.get("manipulation", "")
            step = (latest if latest.startswith(f"{active}:") else active) if active else "no arm command"
            closed = control.manipulation is not None and control.manipulation.gripper_closed
            contacts = self.scene_def.contacts(self.settings["contact_record_n"])
            self.recorder.contacts_tick(runtime.time_seconds, contacts, step, gripper_closed=closed)
        fallen = self.robot.fallen_height
        if fallen is not None and float(self.base.pose()[2]) < fallen:
            runtime.emergency_stop("The robot fell")
        if self._during is not None:
            self._during(self)
        if self.viewer is not None and runtime.tick % self.render_ticks == 0:
            self.viewer.update(self._view())
        if self._record and runtime.tick % self.render_ticks == 0:
            packet = control.pipeline.view("recorder").observe()
            self.recorder.frame(
                packet.cameras["third_person"].rgb.cpu().numpy(),
                title=f"{self.spec.id} | {self.task.goal}: {self.task.description}",
                sim_time=runtime.time_seconds,
                lines=control.narration.lines(),
                spans=control.timeline.spans,
                panels=self._panels.sources(packet),
            )

    def _view_idle(self) -> None:
        """Keep the viewer updated, once per camera period of wall time, while time stands still."""
        if self.viewer is not None and time.perf_counter() - self._last_view >= self.settings["camera_period"]:
            self._last_view = time.perf_counter()
            self.viewer.update(self._view())

    def _view(self) -> EpisodeView:
        """What the viewer may read now (main thread)."""
        control, runtime = self.control, self.control.runtime
        navigation = control.navigation
        costmap = getattr(navigation.backend, "costmap", None) if navigation is not None else None
        log = control.narration.log
        return EpisodeView(
            episode_id=runtime.episode_id,
            time_seconds=runtime.time_seconds,
            status=runtime.status.value,
            observe=control.pipeline.view("viewer").observe,
            costmap=costmap if costmap is not None else (lambda: None),
            narration=lambda start: list(log[start:]),
            holding=control.manipulation is not None and control.manipulation.holding,
        )

    def _render_idle(self) -> None:
        """Keep a Kit window responsive while time stands still."""
        if self.sim.has_gui and time.perf_counter() - self._last_render > 0.1:
            self.sim.render()
            self._last_render = time.perf_counter()

    def truth(self) -> Truth:
        """Ground truth now (robot tests only)."""
        return Truth(self)

    def close(self) -> None:
        if self.planner is not None:
            self.planner.close()
        self.recorder.close()
        self.control.close()
        if self._nav2 is not None:
            self._nav2.close()

    def __enter__(self) -> Environment:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_episode(spec: EpisodeSpec | str | Path, agent: Callable | str, output: Path, **options) -> dict:
    """Build the episode, run the agent, write and return the report (``task`` among the options)."""
    if not isinstance(spec, EpisodeSpec):
        spec = EpisodeSpec.load(spec)
    with Environment(spec, output, **options) as environment:
        return environment.run(agent)


def _look_rotation(eye: np.ndarray, target: np.ndarray) -> list[float]:
    """Camera orientation [qx, qy, qz, qw] (world convention: x forward) looking from eye at target."""
    from scipy.spatial.transform import Rotation

    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    return Rotation.from_matrix(np.column_stack((forward, left, np.cross(forward, left)))).as_quat().tolist()
