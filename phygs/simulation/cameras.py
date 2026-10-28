# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation cameras rendered by RTX and read from Replicator annotators on the GPU."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


@configclass
class RenderCameraCfg(AssetBaseCfg):
    """A camera prim in the scene, read by :class:`RenderCameras` (not an IsaacLab camera sensor).

    ``spawn`` creates a pinhole camera at ``init_state`` relative to the parent prim, in USD's
    camera convention (forward ``-Z``, up ``+Y``); without ``spawn`` the prim must exist, for
    example a camera in the robot asset.
    """

    width: int = MISSING
    """Image width [pixels]."""

    height: int = MISSING
    """Image height [pixels]."""

    data_types: list[str] = ["rgb", "distance_to_image_plane"]
    """Rendered products, from :attr:`RenderCameras.DATA_TYPES`."""


class RenderCameras:
    """RGB and depth images of the scene's :class:`RenderCameraCfg` cameras.

    Cameras with the same resolution and data types share one tiled render product (``tiled``,
    the default; Replicator's ``render_product_tiled``, as IsaacLab's camera sensor): RTX draws
    them into one image in one pass, and after each render the tiles are cut into per-camera
    images on the GPU. In the house, six 640x480 RGB-D cameras took 13.5 ms per frame tiled,
    three took 16.6 ms with a render product each (2026-10-27). ``tiled=False`` gives every camera
    its own render product. Annotators deliver on the GPU when CUDA is available, and each read
    copies one image to the scene device. A camera's pose is its parent body's physics pose times
    the fixed mount from USD (the nearest rigid-body ancestor of the camera prim); a camera with
    no rigid-body ancestor keeps its USD world pose.
    """

    DATA_TYPES = ("rgb", "distance_to_image_plane")
    WARMUP_RENDERS = 10
    """Renders allowed before every annotator delivers its first image."""
    _renders = 0  # Renders so far (every instance's): tiles are cut once per render.

    @dataclass
    class _Group:
        """One render product and its annotators, for ``names`` (tiles in this order)."""

        names: tuple[str, ...]
        product: object
        annotators: dict
        images: dict  # Data type -> [cameras, height, width, channels] tensor of the last cut.
        cut_at: int = -1

    @dataclass(frozen=True)
    class Frame:
        """One camera's image at the last render."""

        rgb: torch.Tensor | None
        """Color, shape [height, width, 3], uint8."""

        depth: torch.Tensor | None
        """Distance to the image plane [m], shape [height, width], float; ``inf`` where nothing is hit.
        On the device the renderer delivers on (the GPU when CUDA is available), so that a reader can
        process it there before copying it to the scene device."""

        intrinsics: torch.Tensor
        """Pinhole matrix [pixels], shape [3, 3]."""

        world_from_camera: torch.Tensor
        """Optical frame (forward ``+Z``, down ``+Y``) in the world [m], shape [4, 4]."""

    def __init__(self, scene: InteractiveScene, names: tuple[str, ...], *, tiled: bool = True) -> None:
        import isaaclab.sim as sim_utils
        import omni.replicator.core as rep
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_rtx_hydra_engine_attached
        from pxr import Usd, UsdGeom

        ensure_rtx_hydra_engine_attached()
        self._scene = scene
        self._device = scene.device
        annotator_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._annotator_device = annotator_device
        self._cameras, self._groups = {}, []
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        members: dict[tuple, list[tuple[str, str]]] = {}
        for name in names:
            cfg = getattr(scene.cfg, name, None)
            if not isinstance(cfg, RenderCameraCfg) or name not in scene.extras:
                raise ValueError(f"Missing render camera: {name}")
            unknown = set(cfg.data_types) - set(self.DATA_TYPES)
            if unknown:
                raise ValueError(f"Camera {name} requests unsupported data types {sorted(unknown)}")
            paths = sim_utils.find_matching_prim_paths(cfg.prim_path)
            if len(paths) != 1:
                raise ValueError(f"Camera {name} must match one prim, found {len(paths)}: {cfg.prim_path}")
            prim = UsdGeom.Camera(scene.stage.GetPrimAtPath(paths[0]))
            if not prim:
                raise ValueError(f"Camera {name} is not a USD camera: {paths[0]}")
            key = (cfg.width, cfg.height, tuple(cfg.data_types)) if tiled else (name,)
            members.setdefault(key, []).append((name, paths[0]))
            # As IsaacLab's camera: square pixels, centered principal point, no aperture offsets.
            focal = prim.GetFocalLengthAttr().Get()
            f_x = cfg.width * focal / prim.GetHorizontalApertureAttr().Get()
            intrinsics = torch.tensor(
                [[f_x, 0.0, cfg.width / 2], [0.0, f_x, cfg.height / 2], [0.0, 0.0, 1.0]], device=self._device
            )
            body, mount = self._mount(scene, prim.GetPrim(), xforms)
            self._cameras[name] = [cfg, None, 0, intrinsics, body, mount]  # Group and tile set below.
        for group in members.values():
            cfg = self._cameras[group[0][0]][0]
            paths = [path for _, path in group]
            if tiled:
                product = rep.create.render_product_tiled(cameras=paths, tile_resolution=(cfg.width, cfg.height))
            else:
                product = rep.create.render_product(paths[0], (cfg.width, cfg.height))
            annotators = {}
            for data_type in cfg.data_types:
                annotator = rep.AnnotatorRegistry.get_annotator(data_type, device=annotator_device, do_array_copy=False)
                annotator.attach([product.path])
                annotators[data_type] = annotator
            index = len(self._groups)
            self._groups.append(self._Group(tuple(name for name, _ in group), product, annotators, {}))
            for tile, (name, _) in enumerate(group):
                self._cameras[name][1:3] = [index, tile]
        self._disable_unused_viewport()
        for _ in range(self.WARMUP_RENDERS):
            if all(self._ready(name) for name in self._cameras):
                break
            self.render()
        else:
            raise RuntimeError("Render cameras delivered no images after warm-up renders.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._cameras)

    def data_types(self, name: str) -> tuple[str, ...]:
        return tuple(self._cameras[name][0].data_types)

    @staticmethod
    def render() -> None:
        """Render every camera once at the current physics state."""
        import isaaclab.sim as sim_utils
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        sim_utils.SimulationContext.instance().render()
        # IsaacLab's camera update: physics results to Fabric, then one Kit app update.
        ensure_isaac_rtx_render_update()
        RenderCameras._renders += 1

    def read(self, name: str) -> Frame:
        """Return copies of the camera's images at the last render, with its pose and intrinsics: RGB
        on the scene device, depth on the renderer's device (:attr:`Frame.depth`)."""
        from isaaclab.utils.math import matrix_from_quat

        cfg, index, tile, intrinsics, body, mount = self._cameras[name]
        images = self._images(self._groups[index])
        rgb = depth = None
        if "rgb" in images:
            rgb = images["rgb"][tile, ..., :3].to(self._device, copy=True)
        if "distance_to_image_plane" in images:
            depth = images["distance_to_image_plane"][tile, ..., 0].clone()
        world_from_camera = mount.clone()
        if body is not None:
            asset, body_index = body
            pose = (
                asset.data.body_link_pose_w.torch[0, body_index]
                if body_index is not None
                else asset.data.root_link_pose_w.torch[0]
            )
            world_from_body = torch.eye(4, device=self._device)
            world_from_body[:3, :3] = matrix_from_quat(pose[3:].unsqueeze(0))[0].to(self._device)
            world_from_body[:3, 3] = pose[:3].to(self._device)
            world_from_camera = world_from_body @ mount
        return self.Frame(rgb, depth, intrinsics.clone(), world_from_camera)

    def _images(self, group: _Group) -> dict[str, torch.Tensor]:
        """The group's images of the last render, [cameras, height, width, channels]: its tiles cut
        apart on the renderer's device, once per render (a GPU kernel; under 0.1 ms for six)."""
        import warp as wp
        from isaaclab.utils.warp.kernels import reshape_tiled_image

        if group.cut_at == RenderCameras._renders and group.images:
            return group.images
        cfg = self._cameras[group.names[0]][0]
        count, columns = len(group.names), math.ceil(math.sqrt(len(group.names)))
        for data_type, annotator in group.annotators.items():
            data = annotator.get_data()
            buffer = data if isinstance(data, wp.array) else wp.array(np.asarray(data), device=self._annotator_device)
            channels = 4 if data_type == "rgb" else 1
            out = group.images.get(data_type)
            if out is None:
                dtype = torch.uint8 if data_type == "rgb" else torch.float32
                out = torch.zeros((count, cfg.height, cfg.width, channels), dtype=dtype, device=self._annotator_device)
                group.images[data_type] = out
            wp.launch(
                kernel=reshape_tiled_image,
                dim=(count, cfg.height, cfg.width),
                inputs=[buffer.flatten(), wp.from_torch(out), cfg.height, cfg.width, channels, columns],
                device=str(out.device) if out.is_cuda else "cpu",
            )
        group.cut_at = RenderCameras._renders
        return group.images

    def close(self) -> None:
        """Detach annotators and destroy the render products."""
        for group in self._groups:
            for annotator in group.annotators.values():
                annotator.detach([group.product.path])
            group.product.destroy()
        self._groups.clear()
        self._cameras.clear()

    def _mount(self, scene: InteractiveScene, prim, xforms) -> tuple[tuple | None, torch.Tensor]:
        """Return the camera's parent body (asset, body index or ``None`` for the root) and the
        optical frame in that body's frame, or ``None`` and the optical frame in the world."""
        import isaaclab.sim as sim_utils
        import numpy as np
        from pxr import UsdPhysics

        # USD cameras look along -Z with +Y up; the optical frame looks along +Z with +Y down.
        optical = np.diag([1.0, -1.0, -1.0, 1.0])

        def world(p):
            matrix = np.array(xforms.GetLocalToWorldTransform(p)).T  # USD multiplies row vectors.
            matrix[:3, :3] /= np.linalg.norm(matrix[:3, :3], axis=0)  # Assets may carry a scale.
            return matrix

        link = prim.GetParent()
        while link and not link.IsPseudoRoot() and not link.HasAPI(UsdPhysics.RigidBodyAPI):
            link = link.GetParent()
        if not link or link.IsPseudoRoot():
            return None, torch.tensor(world(prim) @ optical, dtype=torch.float32, device=self._device)
        mount = torch.tensor(
            np.linalg.inv(world(link)) @ world(prim) @ optical, dtype=torch.float32, device=self._device
        )
        link_path = str(link.GetPath())
        for asset in scene.articulations.values():
            root = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)[0]
            if link_path.startswith(root + "/") and link.GetName() in asset.body_names:
                return (asset, asset.body_names.index(link.GetName())), mount
        for asset in scene.rigid_objects.values():
            if link_path == sim_utils.find_matching_prim_paths(asset.cfg.prim_path)[0]:
                return (asset, None), mount
        raise ValueError(f"Camera {prim.GetPath()} is mounted on an unregistered body: {link_path}")

    def _ready(self, name: str) -> bool:
        annotators = self._groups[self._cameras[name][1]].annotators
        return all(getattr(annotator.get_data(), "size", 0) > 0 for annotator in annotators.values())

    @staticmethod
    def _disable_unused_viewport() -> None:
        """Stop rendering Kit's default viewport when no window or visualizer shows it."""
        import isaaclab.sim as sim_utils

        sim = sim_utils.SimulationContext.instance()
        if sim.has_gui or sim.has_active_visualizers():
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport
        except ImportError:  # No viewport extension: nothing renders it.
            return
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.updates_enabled = False


class ViewCamera:
    """A third-person camera for videos, added to a running scene (no scene config entry).

    A USD pinhole camera at ``path``; :meth:`capture` places it at ``eye`` looking at ``target``
    (world [m]), renders once and returns the RGB image. Moving it every frame makes a follow
    camera. Rendering costs wall time; physics is not touched.
    """

    def __init__(self, path: str = "/World/ViewCamera", *, width: int = 800, height: int = 450) -> None:
        import omni.replicator.core as rep
        import omni.usd
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_rtx_hydra_engine_attached
        from pxr import UsdGeom

        ensure_rtx_hydra_engine_attached()
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * height / width)
        camera.CreateClippingRangeAttr((0.05, 100.0))
        self._transform = UsdGeom.Xformable(camera).AddTransformOp()
        self._product = rep.create.render_product(path, (width, height))
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annotator.attach([self._product.path])

    def capture(self, eye, target) -> np.ndarray:
        """RGB image [height, width, 3] uint8 from ``eye`` looking at ``target`` (world [m])."""
        import numpy as np
        from pxr import Gf

        eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
        forward = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # USD cameras look along -Z with +Y up; Gf matrices hold the basis vectors as rows.
        rows = (*right, 0.0), (*up, 0.0), (*(-forward), 0.0), (*eye, 1.0)
        self._transform.Set(Gf.Matrix4d(*[value for row in rows for value in row]))
        for _ in range(RenderCameras.WARMUP_RENDERS):
            RenderCameras.render()
            data = self._annotator.get_data()
            if getattr(data, "size", 0) > 0:
                return np.asarray(data)[..., :3]
        raise RuntimeError("The view camera delivered no image.")

    def close(self) -> None:
        self._annotator.detach([self._product.path])
        self._product.destroy()
