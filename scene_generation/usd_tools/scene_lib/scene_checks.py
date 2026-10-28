# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation diagnostics for residential scenes; never alter states to pass checks."""

from __future__ import annotations

import time

import numpy as np
import torch

import omni.physics.tensors as tensors
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage_id


class SceneChecks:
    """Check rest stability and force-driven joint motion against authored poses."""

    def __init__(self, sim: SimulationContext):
        self.sim = sim
        self.view = tensors.create_simulation_view("torch", stage_id=get_current_stage_id())
        self.bodies = self.view.create_rigid_body_view("/World/objects/*")
        if self.bodies.count == 0:
            raise RuntimeError("Scene validation requires rigid bodies under /World/objects.")
        self.paths = list(self.bodies.prim_paths)
        self.initial = self.bodies.get_transforms().cpu().numpy().copy()
        self.started = time.perf_counter()
        self.stable_samples = 0
        self.settled_seconds: float | None = None
        self.samples = []
        self.stored_geometry = {}
        self.content_results = []
        self.storage_bounds = {}
        bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], False, True)
        transforms = UsdGeom.XformCache()
        for i, path in enumerate(self.paths):
            body = sim.stage.GetPrimAtPath(path)
            if not body.GetCustomDataByKey("support"):
                continue
            inverse = transforms.GetLocalToWorldTransform(body).GetInverse()
            points = []
            for prim in Usd.PrimRange(body):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                transform = transforms.GetLocalToWorldTransform(prim) * inverse
                if prim.IsA(UsdGeom.Cube):
                    half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
                    local = np.array([(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)])
                elif prim.IsA(UsdGeom.Mesh):
                    local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                else:
                    raise ValueError(f"Stored-object contact check does not support collider {prim.GetPath()}")
                matrix = np.asarray(transform)
                points.append(local @ matrix[:3, :3] + matrix[3, :3])
            if not points:
                raise ValueError(f"Stored object has no collision geometry: {path}")
            self.stored_geometry[i] = np.concatenate(points)
            support = body.GetCustomDataByKey("support")
            if body.GetCustomDataByKey("compartment") or support in (
                "/World/objects/kitchen_oven", "/World/objects/kitchen_microwave"
            ):
                support = support if support.startswith("/") else "/World/objects/" + support
                container = sim.stage.GetPrimAtPath(support)
                box = bounds.ComputeUntransformedBound(container).ComputeAlignedRange()
                self.storage_bounds[i] = (self.paths.index(support), np.array(box.GetMin()), np.array(box.GetMax()))


    def check_containment(self, poses: np.ndarray) -> None:
        """Check stored collision geometry remains inside its moving container [m]."""
        for index, (support, low, high) in self.storage_bounds.items():
            rotations = []
            for body in (index, support):
                x, y, z, w = map(float, poses[body, 3:])
                rotations.append(np.asarray(Gf.Matrix3d(Gf.Quatd(w, Gf.Vec3d(x, y, z)))))
            world = self.stored_geometry[index] @ rotations[0] + poses[index, :3]
            local = (world - poses[support, :3]) @ rotations[1].T
            if np.any(local.min(0) < low - 0.015) or np.any(local.max(0) > high + 0.015):
                raise AssertionError(f"Contents escaped container: {self.paths[index]} in {self.paths[support]}")

    def stored_bottom(self, index: int, pose: np.ndarray) -> float:
        """Return the lowest actual collision vertex in world coordinates [m]."""
        x, y, z, w = pose[3:]
        vertical = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
        points = self.stored_geometry[index]
        return float(
            (points[:, 0] * vertical[0] + points[:, 1] * vertical[1] + points[:, 2] * vertical[2]).min() + pose[2]
        )

    def sample(self, simulated_seconds: float) -> None:
        """Check position drift [m] and linear/angular speeds [m/s, rad/s]."""
        poses = self.bodies.get_transforms().cpu().numpy().copy()
        velocities = self.bodies.get_velocities().cpu().numpy().copy()
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise AssertionError("Non-finite rigid-body state.")
        drift = np.linalg.norm(poses[:, :3] - self.initial[:, :3], axis=1)
        angle = 2 * np.arccos(np.clip(np.abs(np.sum(poses[:, 3:] * self.initial[:, 3:], axis=1)), 0, 1))
        speed = np.linalg.norm(velocities[:, :3], axis=1)
        spin = np.linalg.norm(velocities[:, 3:], axis=1)
        worst = int(np.argmax(drift))
        sample = dict(
            simulated_seconds=simulated_seconds,
            max_drift_m=float(drift.max()),
            max_angle_rad=float(angle.max()),
            max_speed_m_s=float(speed.max()),
            max_spin_rad_s=float(spin.max()),
            worst_body=self.paths[worst],
            fastest_body=self.paths[int(np.argmax(speed))],
            fastest_rotating_body=self.paths[int(np.argmax(spin))],
        )
        self.samples.append(sample)
        print(f"[STABILITY] {sample}", flush=True)
        if drift.max() > 0.03 or angle.max() > 0.035:
            failures = [
                (path, self.initial[i].tolist(), poses[i].tolist(), velocities[i].tolist())
                for i, path in enumerate(self.paths)
                if drift[i] > 0.01 or angle[i] > 0.02
            ]
            print("[FAILURE STATES]", failures, flush=True)
            raise AssertionError(f"Uncommanded body movement exceeded 3 cm / 2 degrees: {sample}")
        stable = speed.max() < 0.01 and spin.max() < 0.02
        self.stable_samples = self.stable_samples + 1 if stable else 0
        if self.stable_samples >= 3 and self.settled_seconds is None:
            self.settled_seconds = time.perf_counter() - self.started
        if time.perf_counter() - self.started > 300 and self.settled_seconds is None:
            raise AssertionError("Scene did not settle within 300 wall-clock seconds.")

    def exercise_joints(self, paths: list[str] | None = None) -> list[dict]:
        """Open and close every passive joint using forces [N] and torques [N·m]."""
        joints = [
            p for p in self.sim.stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)
        ]
        if paths is not None:
            requested = set(paths)
            joints = [prim for prim in joints if str(prim.GetPath()) in requested]
            missing = requested - {str(prim.GetPath()) for prim in joints}
            if missing:
                raise ValueError(f"Unknown movable joints: {sorted(missing)}")
        if not joints:
            raise AssertionError("No movable joints to exercise.")
        forces = np.zeros((self.bodies.count, 3), dtype=np.float32)
        torques = np.zeros_like(forces)
        records = []
        for prim in joints:
            joint = UsdPhysics.Joint(prim)
            path = str(joint.GetBody1Rel().GetTargets()[0])
            index = self.paths.index(path)
            sliding = prim.IsA(UsdPhysics.PrismaticJoint)
            rotation = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
            axis_name = prim.GetAttribute("physics:axis").Get()
            local_axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis_name]
            axis = np.array(rotation.TransformDir(Gf.Vec3d(*local_axis)))
            if sliding:
                forces[index] = axis * 12
            else:
                torques[index] = axis * (
                    20 if UsdPhysics.MassAPI(self.sim.stage.GetPrimAtPath(path)).GetMassAttr().Get() > 10 else 3
                )
            records.append(
                dict(
                    joint=str(prim.GetPath()),
                    body=path,
                    index=index,
                    sliding=sliding,
                    force_N=forces[index].tolist(),
                    torque_Nm=torques[index].tolist(),
                    axis=axis.tolist(),
                    lower_limit=float(prim.GetAttribute("physics:lowerLimit").Get()),
                    upper_limit=float(prim.GetAttribute("physics:upperLimit").Get()),
                )
            )
        indices = torch.tensor([record["index"] for record in records], dtype=torch.int32, device=self.sim.device)
        start = self.bodies.get_transforms().cpu().numpy().copy()
        results = []
        forces_tensor = torch.as_tensor(forces, device=self.sim.device)
        torques_tensor = torch.as_tensor(torques, device=self.sim.device)
        for direction in (1, -1):
            started = time.perf_counter()
            applied_forces = direction * forces_tensor
            applied_torques = direction * torques_tensor
            for step in range(round(5 / self.sim.cfg.dt)):
                self.bodies.apply_forces_and_torques_at_position(applied_forces, applied_torques, None, indices, True)
                self.sim.step(render=False)
                if (step + 1) % round(1 / self.sim.cfg.dt) == 0:
                    print(f"[JOINT_PROGRESS] direction={direction}, seconds={(step + 1) * self.sim.cfg.dt:.1f}, "
                          f"wall_seconds={time.perf_counter() - started:.2f}", flush=True)
                if step % round(0.1 / self.sim.cfg.dt) == 0:
                    states = self.bodies.get_transforms().cpu().numpy()
                    if not np.isfinite(states).all():
                        raise AssertionError("Non-finite state during interaction.")
                    for i in self.stored_geometry:
                        clearance = self.stored_bottom(i, states[i]) - self.stored_bottom(i, start[i])
                        if clearance < -0.015:
                            raise AssertionError(
                                f"Contents fell below their support: {self.paths[i]}; "
                                f"step={step}; clearance={clearance}; pose={states[i].tolist()}"
                            )
            poses = self.bodies.get_transforms().cpu().numpy().copy()
            failures = []
            for record in records:
                i = record["index"]
                if record["sliding"]:
                    axis = forces[i] / np.linalg.norm(forces[i])
                    value = float(np.dot(poses[i, :3] - start[i, :3], axis))
                    lower, upper, minimum = (
                        record["lower_limit"] - 0.005,
                        record["upper_limit"] + 0.005,
                        record["upper_limit"] * 0.75,
                    )
                    unit = "m"
                else:
                    current, initial = poses[i, 3:], start[i, 3:]
                    vector = initial[3] * current[:3] - current[3] * initial[:3] - np.cross(current[:3], initial[:3])
                    scalar = float(np.dot(current, initial))
                    value = float(2 * np.arctan2(np.dot(vector, record["axis"]), scalar))
                    lower = np.deg2rad(record["lower_limit"] - 1)
                    upper = np.deg2rad(record["upper_limit"] + 1)
                    minimum = np.deg2rad(record["upper_limit"] * 0.65)
                    unit = "rad"
                result = {**record, "direction": direction, "position": value, "unit": unit}
                results.append(result)
                print(f"[JOINT] {result}", flush=True)
                if not lower <= value <= upper:
                    failures.append(("Joint escaped limits", result))
                if direction == 1 and value < minimum:
                    failures.append(("Joint failed to open", result))
                if direction == -1 and abs(value) > (0.015 if record["sliding"] else 0.05):
                    failures.append(("Joint failed to close", result))
            if failures:
                raise AssertionError(f"Joint interaction failures: {failures}")
            # Body origins move when loose contents tip. Contact geometry measures retained support.
            for i in self.stored_geometry:
                clearance = self.stored_bottom(i, poses[i]) - self.stored_bottom(i, start[i])
                if abs(clearance) > 0.015:
                    raise AssertionError(f"Contents did not retain support: {self.paths[i]}; clearance={clearance}")
        stable_samples = 0
        for second in range(1, 11):
            for _ in range(round(1 / self.sim.cfg.dt)):
                self.sim.step(render=False)
            velocities = self.bodies.get_velocities().cpu().numpy()
            contents = velocities[list(self.stored_geometry)]
            speed = float(np.linalg.norm(contents[:, :3], axis=1).max())
            spin = float(np.linalg.norm(contents[:, 3:], axis=1).max())
            print(f"[CONTENTS_SETTLING] seconds={second}, max_speed_m_s={speed}, max_spin_rad_s={spin}", flush=True)
            stable_samples = stable_samples + 1 if speed <= 0.01 and spin <= 0.02 else 0
            if stable_samples == 2:
                break
        if stable_samples < 2:
            raise AssertionError("Contents did not remain stable for two samples within ten seconds after interaction")
        poses = self.bodies.get_transforms().cpu().numpy().copy()
        velocities = self.bodies.get_velocities().cpu().numpy().copy()
        self.check_containment(poses)
        for i in self.stored_geometry:
            result = dict(
                body=self.paths[i],
                support_error_m=self.stored_bottom(i, poses[i]) - self.stored_bottom(i, start[i]),
                speed_m_s=float(np.linalg.norm(velocities[i, :3])),
                spin_rad_s=float(np.linalg.norm(velocities[i, 3:])),
            )
            self.content_results.append(result)
            if abs(result["support_error_m"]) > 0.015 or result["speed_m_s"] > 0.01 or result["spin_rad_s"] > 0.02:
                raise AssertionError(f"Contents did not settle after interaction: {result}")
        for path, pose, velocity in zip(
            self.paths, poses.tolist(), velocities.tolist()
        ):
            print(f"[STATE] body={path}, pose={pose}, velocity={velocity}", flush=True)
        return results

    def drop_objects(self, paths: list[str] | None = None) -> list[dict]:
        """Lift loose objects with forces [N], release, and measure natural landing [m]."""
        if paths is None:
            paths = [f"/World/objects/{name}" for name in ("coffee_book", "shelf_book", "kitchen_drawers_contents")]
        selected = [self.paths.index(path) for path in paths]
        indices = torch.tensor(selected, dtype=torch.int32, device=self.sim.device)
        forces = torch.zeros((self.bodies.count, 3), device=self.sim.device)
        masses = self.bodies.get_masses().flatten()
        for i in selected:
            forces[i, 2] = 1.5 * 9.81 * masses[i]
        start = self.bodies.get_transforms().cpu().numpy().copy()
        peak = start[:, 2].copy()
        max_speed = np.zeros(self.bodies.count)
        max_spin = np.zeros(self.bodies.count)
        lift_steps = round(0.2 / self.sim.cfg.dt)
        total_steps = round(6 / self.sim.cfg.dt)
        for step in range(total_steps):
            if step < lift_steps:
                self.bodies.apply_forces_and_torques_at_position(forces, None, None, indices, True)
            self.sim.step(render=False)
            poses = self.bodies.get_transforms().cpu().numpy().copy()
            velocities = self.bodies.get_velocities().cpu().numpy().copy()
            if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
                raise AssertionError("Non-finite body state in drop test.")
            peak = np.maximum(peak, poses[:, 2])
            if step >= round(5 / self.sim.cfg.dt):
                max_speed = np.maximum(max_speed, np.linalg.norm(velocities[:, :3], axis=1))
                max_spin = np.maximum(max_spin, np.linalg.norm(velocities[:, 3:], axis=1))
        results = []
        for i in selected:
            result = dict(
                body=self.paths[i],
                applied_force_N=float(forces[i, 2]),
                lift_seconds=0.2,
                rise_m=float(peak[i] - start[i, 2]),
                landing_error_m=float(abs(poses[i, 2] - start[i, 2])),
                horizontal_drift_m=float(np.linalg.norm(poses[i, :2] - start[i, :2])),
                final_second_max_speed_m_s=float(max_speed[i]),
                final_second_max_spin_rad_s=float(max_spin[i]),
            )
            print(f"[DROP] {result}", flush=True)
            results.append(result)
            if result["rise_m"] < 0.05 or result["landing_error_m"] > 0.005:
                raise AssertionError(f"Object failed lift-and-land check: {result}")
            if result["horizontal_drift_m"] > 0.02 or max_speed[i] > 0.01 or max_spin[i] > 0.03:
                raise AssertionError(f"Object did not settle after landing: {result}")
        return results
