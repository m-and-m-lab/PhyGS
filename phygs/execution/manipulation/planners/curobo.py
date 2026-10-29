# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""cuRobo V2 adapter, qualified against commit 78fd485fa82d9b9a063fb4985e371814587e666a."""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from curobo._src.collision.attachment_manager import AttachmentManager  # Not exported; the held object.
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.perception import Mapper, MapperCfg, RobotSegmenter
from curobo.robot_builder import RobotBuilder
from curobo.scene import Cuboid, Scene
from curobo.sphere_fit import SphereFitType
from curobo.types import CameraObservation, GoalToolPose, JointState, Pose

from phygs.execution.manipulation.planners.interface import (
    DepthImage,
    HeldShape,
    IkRequest,
    IkResult,
    PlanRequest,
    PlanResult,
)
from phygs.execution.manipulation.planners.model import gripper_positions

HELD_LINK, HELD_SPHERES = "held_object", 256  # Reserved wrist link and its sphere slots for a held object.
SPHERE_FIT = dict(sphere_density=1.0, coverage_weight=1.0e5)  # MorphIt volume fit, ~260 spheres in all.
# The robot's thin links (the model's surface_fit_links, e.g. Spot's jaws and wrist): 8 mm spheres on
# the mesh surface (Spot's reach 0.8 cm past the mesh, where the volume fit reached 1.7-2.7 cm).
SURFACE_FIT = dict(sphere_density=2.0, surface_radius=0.008)
CACHE = Path.home() / ".cache/phygs/curobo"
# Box slots per planner; the collision kernel scans every slot, so keep it small. 64: the episodes'
# obstacles are the scene's colliders around the robot (36 at the house's kitchen island).
CUBOIDS = 64
GOALSET = 8  # Most candidate goal poses in one request.
POSTURE_TOLERANCE = 1e-3  # Largest leg angle change [rad] a planner keeps its posture for.


def robot_config(config: dict) -> dict:
    """cuRobo robot config from the planning URDF: spheres fitted by RobotBuilder, cached by content.

    The collision matrix adds the model's skip list (adjacent links and links the arm does not
    move), because the legs stand in a measured posture, not cuRobo's default one.
    """
    urdf = Path(config["urdf"])
    # The collision matrix is baked into the cache, so the skip list is part of the key.
    surface_links = config["surface_fit_links"]
    fit = json.dumps([SPHERE_FIT, surface_links, SURFACE_FIT, config["self_collision_ignore"]], sort_keys=True)
    key = hashlib.sha256((urdf.read_text() + fit).encode()).hexdigest()[:16]
    cached = CACHE / f"{config['name']}_{key}.yml"
    # The planning URDF uses file:// mesh URLs for MoveIt; cuRobo reads plain paths.
    plain = urdf.with_name(urdf.stem + "_plain_paths.urdf")
    plain.write_text(urdf.read_text().replace('filename="file://', 'filename="'))
    if not cached.exists():
        builder = RobotBuilder(str(plain), tool_frames=[config["wrist_link"]])
        builder.fit_collision_spheres(use_collision_mesh=True, **SPHERE_FIT)
        for link in surface_links:
            builder.refit_link_spheres(link, use_collision_mesh=True, fit_type=SphereFitType.SURFACE, **SURFACE_FIT)
        builder.compute_collision_matrix(custom_ignore=config["self_collision_ignore"])
        CACHE.mkdir(parents=True, exist_ok=True)
        builder.save(builder.build(), str(cached))
    kinematics = yaml.safe_load(cached.read_text())["kinematics"]
    locked = config["locked_joints"]
    kinematics.update(
        urdf_path=str(plain),
        lock_joints=dict(locked),
        collision_link_names=[*kinematics["collision_link_names"], HELD_LINK],
        extra_links={
            HELD_LINK: dict(
                parent_link_name=config["wrist_link"],
                link_name=HELD_LINK,
                joint_name="held_object_joint",
                joint_type="FIXED",
                fixed_transform=[0, 0, 0, 1, 0, 0, 0],
            )
        },
        extra_collision_spheres={HELD_LINK: HELD_SPHERES},
        cspace=dict(
            joint_names=[*config["joints"], *locked],
            default_joint_position=[*config["default_joints"], *locked.values()],
            max_acceleration=config["max_acceleration"],
            max_jerk=100.0,
            velocity_scale=1.0,
            cspace_distance_weight=[1.0] * (len(config["joints"]) + len(locked)),
            null_space_weight=[1.0] * (len(config["joints"]) + len(locked)),
        ),
    )
    kinematics["self_collision_ignore"][HELD_LINK] = list(config["held_contact_links"])
    return {"robot_cfg": {"kinematics": kinematics}}


def cuboid(name: str, box: dict, clearance: float) -> Cuboid:
    """cuRobo box [m] from a pose [x, y, z, qx, qy, qz, qw], grown by clearance per side; cuRobo orders wxyz."""
    dims = [size + 2 * clearance for size in box["size"]]
    return Cuboid(name=name, dims=dims, pose=[*box["pose"][:3], box["pose"][6], *box["pose"][3:6]])


class Planner:
    def __init__(self, config: dict):
        self.config = config
        self.names = config["joints"]
        self.dt = 0.02
        self.robot = robot_config(config)
        self.gripper = config["gripper_joint"]
        grid = config["depth_map"]
        self.mapper = Mapper(
            MapperCfg(
                extent_meters_xyz=tuple(grid["extent"]),
                extent_esdf_meters_xyz=tuple(grid["extent"]),
                voxel_size=grid["voxel_size"] / 2,  # TSDF; the ESDF for collision uses voxel_size.
                esdf_voxel_size=grid["voxel_size"],
                grid_center=torch.tensor(grid["center"], dtype=torch.float32),
                # Two ESDF voxels: one image per camera (each plan builds a new map) left gaps in a
                # surface seen at a grazing angle with one voxel; a held box passed through the
                # clutter's top (2026-10-23, offline on the planning test's images).
                truncation_distance=2 * grid["voxel_size"],
                num_cameras=1,
                image_height=grid["image_shape"][0],  # Every depth image has this shape.
                image_width=grid["image_shape"][1],
                device="cuda",
                block_size=2,
            )
        )
        self.esdf = self.mapper.compute_esdf()
        self.mapped = False  # The ESDF holds a request's images (an empty map otherwise).
        self.planners: dict[float, MotionPlanner] = {}
        self.segmenters: dict[float, tuple[RobotSegmenter, AttachmentManager]] = {}
        gripper = config["gripper_ratios"]
        self.locked = {name: value for name, value in config["locked_joints"].items() if name not in gripper}
        self.posed: dict[float, dict[str, float]] = {}  # The leg angles [rad] each planner holds.
        self.planner_for(config["locked_joints"][self.gripper])

    def planner_for(self, gripper: float) -> MotionPlanner:
        """Warmed-up planner with the gripper locked at gripper [rad]; cuRobo fixes locked joints at build.

        Every planner has the same collision types (boxes and the depth map). A planner with
        other types makes Warp rebuild the collision kernels, and the voxel check of planners
        built before it then fails silently inside their CUDA graphs. Building takes about 15 s.
        """
        angle = round(gripper, 2)
        grid = self.config["depth_map"]
        if angle not in self.planners:
            self.robot["robot_cfg"]["kinematics"]["lock_joints"].update(
                {**self.locked, **gripper_positions(self.config, angle)}
            )
            voxels = {"layers": 1, "dims": self.esdf.dims, "voxel_size": self.esdf.voxel_size}
            planner = MotionPlanner(
                MotionPlannerCfg.create(
                    robot=self.robot,
                    collision_cache={"cuboid": CUBOIDS, "voxel": voxels},
                    self_collision_check=True,
                    interpolation_dt=self.dt,
                    random_seed=7,
                    max_goalset=GOALSET,
                )
            )
            planner.warmup(num_warmup_iterations=1)
            self.planners[angle] = planner
            # The self filter has its own copy of the kinematics (legs, gripper and held object kept
            # equal to the planner's): run on the planner's own, it made the planner miss obstacles.
            # float32: the default bfloat16 fails this commit's own sphere dtype check.
            segmenter = RobotSegmenter(
                Kinematics(KinematicsCfg.from_data_dict(copy.deepcopy(self.robot))),
                distance_threshold=grid["self_filter_padding"],
                use_cuda_graph=False,
                ops_dtype=torch.float32,
            )
            self.segmenters[angle] = segmenter, AttachmentManager(segmenter.kinematics)
            self.posed[angle] = dict(self.locked)
        return self.planners[angle]

    def repose(self, locked: dict[str, float] | None, gripper: float) -> None:
        """Take the measured leg angles [rad] (None keeps the last ones) into the planner for gripper [rad].

        cuRobo bakes locked joints into fixed link transforms. The planner's solvers share one
        kinematics parameter set, so new transforms copied into it move the legs in place: about
        0.1 s, where building a planner takes about 15 s. Only the transforms and the locked
        state change; the collision spheres stay in their link frames.
        """
        if locked is not None:
            if set(locked) != set(self.locked):
                raise ValueError(f"Locked joints {sorted(locked)} do not match the model's {sorted(self.locked)}.")
            self.locked = dict(locked)
        angle = round(gripper, 2)
        moved = (abs(self.posed[angle][name] - value) for name, value in self.locked.items())
        if max(moved, default=0.0) <= POSTURE_TOLERANCE:  # A robot may lock nothing but the gripper.
            return
        robot = copy.deepcopy(self.robot)
        kinematics = robot["robot_cfg"]["kinematics"]
        kinematics["lock_joints"].update({**self.locked, **gripper_positions(self.config, angle)})
        # Transforms only (the self-collision pairs take seconds); the collision links stay in the
        # kinematic tree as tool frames, so the tree and its transforms keep their layout.
        kinematics["tool_frames"] = [*kinematics["tool_frames"], *kinematics["collision_link_names"]]
        kinematics["collision_link_names"] = []
        posed = KinematicsCfg.from_data_dict(robot).kinematics_config
        for model in (self.planners[angle].kinematics, self.segmenters[angle][0].kinematics):
            current = model.config.kinematics_config
            current.fixed_transforms.copy_(posed.fixed_transforms)
            current.lock_jointstate.copy_(posed.lock_jointstate)
        self.posed[angle] = dict(self.locked)

    def state(self, values):
        return JointState.from_position(
            torch.tensor([values], device="cuda", dtype=torch.float32), joint_names=self.names
        )

    def goal(self, planner: MotionPlanner, poses: list[list[float]]) -> GoalToolPose:
        """Wrist goal set from poses [x, y, z, qx, qy, qz, qw]; cuRobo orders quaternions wxyz."""
        if len(poses) > GOALSET:
            raise ValueError(f"The request has {len(poses)} goal poses; the planner takes {GOALSET}.")
        poses = torch.tensor(poses, device="cuda", dtype=torch.float32)
        return GoalToolPose(
            tool_frames=planner.tool_frames,
            position=poses[:, :3].reshape(1, 1, 1, -1, 3),
            quaternion=poses[:, [6, 3, 4, 5]].reshape(1, 1, 1, -1, 4),
        )

    def scene(
        self,
        gripper: float,
        boxes: list[dict],
        held: HeldShape | None,
        start: list[float],
        locked: dict[str, float] | None = None,
        images: list[DepthImage] | None = None,
    ):
        """Planner and start state with the gripper at gripper [rad], the legs at locked [rad] (None
        keeps the last ones), the held object, the boxes and a depth map of the images; returns the
        planner, the start state and the map time [s]."""
        if len(boxes) > CUBOIDS:
            raise ValueError(f"The request has {len(boxes)} boxes; the planner has {CUBOIDS} slots.")
        planner = self.planner_for(gripper)
        self.repose(locked, gripper)
        state = self.state(start)
        segmenter, attachments = self.segmenters[round(gripper, 2)]
        for manager in (planner.attachment_manager, attachments):
            manager.detach(HELD_LINK)
            if held is not None:  # Held object in the wrist frame; collision model only.
                spheres = torch.tensor(held.spheres, device="cuda", dtype=torch.float32)
                if len(spheres) > HELD_SPHERES:
                    raise ValueError(f"The held object needs {len(spheres)} spheres; the model has {HELD_SPHERES}.")
                manager.update(spheres, state, link_name=HELD_LINK)
        mapped = time.perf_counter()
        if images or self.mapped:
            self.fuse(images or [], segmenter, state)
        mapped = time.perf_counter() - mapped
        cuboids = [cuboid(f"obstacle_{i}", box, self.config["clearance"]) for i, box in enumerate(boxes)]
        planner.update_world(Scene(cuboid=cuboids, voxel=[self.esdf]))
        return planner, state, mapped

    def samples(self, trajectory: JointState, count: int | None = None) -> dict:
        """Timed arm-joint samples [rad, s] from the first count interpolated states."""
        order = [trajectory.joint_names.index(n) for n in self.names]
        positions = trajectory.position.reshape(-1, len(trajectory.joint_names))[:count, order].cpu().tolist()
        return dict(joint_names=self.names, positions=positions, times=[i * self.dt for i in range(len(positions))])

    def plan(self, request: PlanRequest) -> PlanResult:
        started = time.perf_counter()
        gripper = self.config["locked_joints"][self.gripper] if request.gripper is None else request.gripper
        planner, state, mapped = self.scene(
            gripper, request.boxes, request.held, request.start, request.locked_joints, request.images
        )
        if request.world_path is not None:
            self.save_world(request.world_path)
        if request.goal_pose is not None:
            result = planner.plan_pose(self.goal(planner, [request.goal_pose]), state, max_attempts=5)
        else:
            result = planner.plan_cspace(self.state(request.goal), state, max_attempts=5)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if result is None or not result.success.any():
            return {"success": False, "reason": "no_plan_within_budget", "plan_seconds": elapsed, "map_seconds": mapped}
        return dict(
            success=True,
            **self.samples(result.get_interpolated_plan()),
            plan_seconds=elapsed,
            map_seconds=mapped,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        )

    def ik(self, request: IkRequest) -> IkResult:
        """cuRobo IK for the wrist pose in the body frame from the seed joints, free of self-collision.

        The scene is cleared first, so obstacles and the depth map do not affect the solution.
        """
        started = time.perf_counter()
        planner = self.planner_for(self.config["locked_joints"][self.gripper])
        self.repose(None, self.config["locked_joints"][self.gripper])
        planner.clear_scene_cache()
        goal = self.goal(planner, [request.pose])
        result = planner.ik_solver.solve_pose(goal, current_state=self.state(request.seed))
        torch.cuda.synchronize()
        if not bool(result.success.view(-1)[0]):
            return {"success": False, "seconds": time.perf_counter() - started}
        solution = result.js_solution
        order = [solution.joint_names.index(n) for n in self.names]
        joints = solution.position.reshape(-1, len(solution.joint_names))[0, order].cpu().tolist()
        return {"success": True, "joints": joints, "seconds": time.perf_counter() - started}

    def save_world(self, path: str) -> None:
        """Save the depth map's obstacle voxels the plans avoid: body-frame centres [m] of the ESDF cells
        within the clearance of a surface (value <= 0), float32 [N,3]; none without a depth map."""
        if not self.mapped:
            np.save(path, np.zeros((0, 3), dtype=np.float32))
            return
        centres = self.esdf.create_xyzr_tensor(transform_to_origin=True)[:, :3]
        values = self.esdf.feature_tensor.reshape(-1)
        if len(values) != len(centres):
            raise ValueError(f"The ESDF has {len(values)} values for {len(centres)} voxel centres.")
        np.save(path, centres[values.to(centres.device) <= 0.0].cpu().numpy().astype(np.float32))

    def fuse(self, images: list[DepthImage], segmenter: RobotSegmenter, state: JointState) -> None:
        """Clear the Mapper TSDF, fuse the images without their robot and held-object pixels, and
        recompute the ESDF that plans avoid."""
        self.mapper.reset()
        for image in images:
            depth = torch.from_numpy(np.load(image.path)).to("cuda", torch.float32)[None]
            p = image.pose
            observation = CameraObservation(
                depth_image=depth,
                rgb_image=torch.zeros((*depth.shape, 3), dtype=torch.uint8, device="cuda"),  # Geometry only.
                pose=Pose(
                    position=torch.tensor([p[:3]], device="cuda", dtype=torch.float32),
                    quaternion=torch.tensor([[p[6], *p[3:6]]], device="cuda", dtype=torch.float32),
                ),
                intrinsics=torch.tensor([image.intrinsics], device="cuda", dtype=torch.float32),
                depth_to_meter=1.0,  # Depth is in metres; the segmenter's default 0.001 assumes millimetres.
            )
            segmenter.update_camera_projection(observation)  # It keeps the first camera's rays otherwise.
            mask, observation.depth_image = segmenter.get_robot_mask(observation, state)
            name, valid = Path(image.path).name, int((depth > 0).sum())
            print(f"self filter: {int(mask.sum())} of {valid} depth pixels on the robot ({name})", flush=True)
            self.mapper.integrate(observation)
        self.esdf = self.mapper.compute_esdf()
        self.esdf.feature_tensor -= self.config["clearance"]  # Obstacles grow by the clearance.
        self.mapped = bool(images)
        torch.cuda.synchronize()

    def close(self):
        pass
