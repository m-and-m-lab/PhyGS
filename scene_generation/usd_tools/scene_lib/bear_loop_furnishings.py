# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Photo-led furnishing groups and a curated inventory of separately movable possessions."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from scene_lib.bear_loop_layout import BATHROOMS
from scene_lib.residential_surfaces import photographic_material

if TYPE_CHECKING:
    from build_bear_loop_scene import BearLoopHouse


def install_reference_materials(house: BearLoopHouse) -> None:
    """Install photographic stone and set neutral household paint and porcelain finishes."""
    assets = Path(__file__).parent.parent / "assets/bear_loop"
    for source in assets.glob("*.jpg"):
        shutil.copy2(source, house.texture_dir / source.name)
    for name, base, color, roughness in (
        ("stone", "plaster", (0.5, 0.5, 0.5), 0.8),
        ("granite", "tile", (0.5, 0.5, 0.5), 0.25),
        ("carpet", "linen", (0.65, 0.60, 0.50), 0.95),
        ("porcelain", "cream", (0.87, 0.88, 0.86), 0.22),
        ("felt", "linen", (0.48, 0.45, 0.39), 0.95),
        ("walnut", "oak", (0.18, 0.075, 0.03), 0.35),
    ):
        destination = "/World/materials/" + name
        Sdf.CopySpec(
            house.stage.GetRootLayer(), house.materials[base].GetPath(), house.stage.GetRootLayer(), destination
        )
        material = UsdShade.Material(house.stage.GetPrimAtPath(destination))
        for prim in Usd.PrimRange(material.GetPrim()):
            for attr in prim.GetAttributes():
                if attr.HasAuthoredConnections():
                    attr.SetConnections(
                        [p.ReplacePrefix(house.materials[base].GetPath(), destination) for p in attr.GetConnections()]
                    )
        shader = UsdShade.Shader(house.stage.GetPrimAtPath(destination + "/surface"))
        shader.GetInput("diffuseColor").DisconnectSource()
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))
        shader.GetInput("roughness").Set(roughness)
        house.materials[name] = material
    photographic_material(house.stage, house.materials["stone"], "rock_wall_08")
    photographic_material(house.stage, house.materials["granite"], "granite_tile_03")
    photographic_material(house.stage, house.materials["carpet"], "dirty_carpet")
    photographic_material(house.stage, house.materials["tile"], "floor_tiles_04")
    # Paint remains neutral under the explicitly white ceiling fittings.
    shader = UsdShade.Shader(house.stage.GetPrimAtPath("/World/materials/cream/surface"))
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.82, 0.81, 0.78))
    (house.metadata_dir / "additional_material_sources.json").write_text((assets / "sources.json").read_text())
    model_sources = Path(house.library.sources["ph_chair"].GetRootLayer().realPath).parent.parent / "sources.json"
    shutil.copy2(model_sources, house.metadata_dir / "model_sources.json")


def native(
    house: BearLoopHouse,
    name: str,
    asset: str,
    position: tuple,
    size: tuple,
    mass: float,
    support: str = "floor",
    approximation: str = "box",
    yaw: float = 0,
    anchored: bool = False,
    evidence: str = "inferred household possession",
) -> str:
    """Place a textured reference prop at a support-relative base [m], with mass [kg]."""
    path = house.prop(
        name, asset, (position[0], position[1], position[2] + size[2] / 2 + 0.002), size, mass, approximation
    )
    house.pose(name, yaw=yaw)
    prim = house.stage.GetPrimAtPath(path)
    prim.SetCustomDataByKey("support", support)
    prim.SetCustomDataByKey("independently_pickable", not anchored)
    if anchored:
        joint = UsdPhysics.FixedJoint.Define(house.stage, "/World/joints/" + name + "_mount")
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(position[0], position[1], position[2] + size[2] / 2 + 0.002))
        joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
    source = house.reference.GetPrimAtPath(house.meshes[asset])
    fingerprint = hashlib.sha256(np.asarray(UsdGeom.Mesh(source).GetPointsAttr().Get()).tobytes()).hexdigest()
    house.inventory.append(
        dict(
            name=name,
            path=path,
            source=asset,
            geometry_sha256=fingerprint,
            evidence=evidence,
            support=support,
            size_m=list(size),
            mass_kg=mass,
        )
    )
    return path


def kitchen_contents(house: BearLoopHouse) -> None:
    """Populate storage with food, drinkware, dishes and utensils instead of bathroom props."""
    choices = [
        ("mixing_bowl", "BowlFactory_974736__spawn_asset_549778_", (0.21, 0.21, 0.12), 0.4, "sdf"),
        ("grain_packet", "FoodBagFactory_6129003__spawn_asset_9279215_", (0.18, 0.12, 0.055), 0.5, "box"),
        ("pasta_box", "FoodBoxFactory_9173839__spawn_asset_5983640_", (0.13, 0.08, 0.21), 0.45, "box"),
        ("coffee_cup", "CupFactory_9054484__spawn_asset_3849534_", (0.105, 0.085, 0.10), 0.22, "sdf"),
        ("small_bowl", "BowlFactory_974736__spawn_asset_581637_", (0.15, 0.15, 0.085), 0.25, "sdf"),
        ("food_tin", "CanFactory_1526963__spawn_asset_5470950_", (0.075, 0.075, 0.12), 0.4, "convexHull"),
        ("cutlery_knife", "KnifeFactory_9767186__spawn_asset_883869_", (0.24, 0.032, 0.014), 0.08, "box"),
        ("grain_packet", "FoodBagFactory_6129003__spawn_asset_9279215_", (0.18, 0.12, 0.055), 0.5, "box"),
        ("snack_packet", "FoodBagFactory_6193854__spawn_asset_649510_", (0.17, 0.12, 0.06), 0.12, "box"),
        ("wine_bottle", "BottleFactory_1804529__spawn_asset_134498_", (0.075, 0.075, 0.26), 1.05, "convexHull"),
        ("water_bottle", "BottleFactory_1021896__spawn_asset_7412270_", (0.075, 0.075, 0.22), 0.6, "convexHull"),
        ("serving_bowl", "BowlFactory_974736__spawn_asset_6346036_", (0.19, 0.19, 0.11), 0.35, "sdf"),
    ]
    for storage, (name, asset, size, mass, approximation) in zip(house.kitchen_stores, choices, strict=True):
        base = storage["world"]((-0.15, 0, storage["support"]))
        path = native(
            house,
            storage["name"] + "_contents_" + name,
            asset,
            base,
            size,
            mass,
            storage["support_path"],
            approximation,
            yaw=7,
        )
        house.stored_items.append(path)
    secondary = [
        ("h48:bedroom_saucer_plate_0_s0_0", (0.18, 0.18, 0.025), 0.20),
        ("h47:bedroom_coffee_mug_0", (0.11, 0.085, 0.10), 0.24),
        ("h98:bedroom_glass_tumbler_0", (0.08, 0.08, 0.11), 0.19),
        ("h48:bedroom_coffee_mug_0_s0_1", (0.105, 0.085, 0.095), 0.22),
    ]
    for storage, (key, size, mass) in zip(house.kitchen_stores[:4], secondary, strict=True):
        position = storage["world"]((0.17, 0.015, storage["support"]))
        path = house.asset(
            storage["name"] + "_contents_drinkware",
            key,
            position,
            size,
            mass,
            support=storage["support_path"],
            kind="sdf",
        )
        house.stored_items.append(path)
    native(
        house,
        "counter_fruit_bowl",
        "BowlFactory_974736__spawn_asset_681501_",
        (11.05, 9.1, 0.9),
        (0.20, 0.20, 0.12),
        0.4,
        "kitchen_base_0",
        "sdf",
        yaw=-11,
    )
    native(
        house,
        "counter_drink",
        "BottleFactory_1021896__spawn_asset_7412270_",
        (12.05, 9.03, 0.9),
        (0.075, 0.075, 0.22),
        0.6,
        "kitchen_base_1",
        "convexHull",
    )


def fireplace(house: BearLoopHouse, name: str, x: float, y: float, z: float) -> None:
    """Build stone surround, recessed firebox, hearth and timber mantel [m]."""
    root = "/World/structure/" + name
    for suffix, center, size, material in (
        ("left", (x, y - 0.83, z + 1.40), (0.44, 0.44, 2.8), "stone"),
        ("right", (x, y + 0.83, z + 1.40), (0.44, 0.44, 2.8), "stone"),
        ("header", (x, y, z + 2.0), (0.44, 1.22, 1.6), "stone"),
        ("hearth", (x + 0.12, y, z + 0.15), (0.8, 2.2, 0.30), "stone"),
        ("firebox", (x + 0.18, y, z + 0.73), (0.025, 1.22, 0.85), "screen"),
        ("mantel", (x + 0.25, y, z + 1.43), (0.65, 2.30, 0.14), "walnut"),
        ("frame_top", (x + 0.24, y, z + 1.17), (0.06, 1.24, 0.035), "metal"),
        ("frame_bottom", (x + 0.24, y, z + 0.31), (0.06, 1.24, 0.035), "metal"),
    ):
        house.box(root + "/" + suffix, center, size, material)
    house.inventory.append(dict(name=name, category="stone_fireplace", evidence="photos_005_006_065", fixed=True))


def pool_table(house: BearLoopHouse) -> None:
    """Model the photographed billiards group, including open pockets and loose balls [m]."""
    x, y, z = 19.1, 7.45, 3.2
    body = house.body("billiards_table", (x, y, z + 0.002), 300, yaw=90)

    def panel(path: str, center: tuple, size: tuple, material: str) -> None:
        house.box(
            path, (center[0] * 0.85, center[1] * 0.85, center[2]), (size[0] * 0.85, size[1] * 0.85, size[2]), material
        )

    panel(body + "/slate_center", (0, 0, 0.775), (2.60, 1.12, 0.05), "felt")
    for side in (-1, 1):
        label = "left" if side < 0 else "right"
        for i, cx in enumerate((-0.64, 0.64)):
            panel(body + f"/slate_{label}_{i}", (cx, side * 0.6475, 0.775), (1.10, 0.175, 0.05), "felt")
            panel(body + f"/rail_{label}_{i}", (cx, side * 0.79, 0.84), (1.10, 0.11, 0.13), "walnut")
        panel(body + f"/end_{label}", (side * 1.365, 0, 0.84), (0.13, 1.12, 0.13), "walnut")
        for end in (-1, 1):
            tag = label + ("_near" if end < 0 else "_far")
            panel(body + "/leg_" + tag, (side * 1.0, end * 0.53, 0.365), (0.20, 0.20, 0.73), "walnut")
    for i, (px, py) in enumerate(((-1.23, -0.65), (0, -0.65), (1.23, -0.65), (-1.23, 0.65), (0, 0.65), (1.23, 0.65))):
        panel(body + f"/pocket_base_{i}", (px, py, 0.49), (0.15, 0.15, 0.035), "screen")
        for j, (dx, dy, w, d) in enumerate(
            ((-0.085, 0, 0.02, 0.19), (0.085, 0, 0.02, 0.19), (0, -0.085, 0.15, 0.02), (0, 0.085, 0.15, 0.02))
        ):
            panel(body + f"/pocket_{i}_{j}", (px + dx, py + dy, 0.635), (w, d, 0.27), "screen")
    for i, (dx, dy) in enumerate(
        ((-0.65, 0.08), (0.52, 0.14), (0.38, -0.18), (0.08, 0.30), (0.73, -0.12), (-0.12, -0.26))
    ):
        path = house.body(f"billiards_ball_{i}", (x - dy * 0.85, y + dx * 0.85, z + 0.832), 0.17)
        sphere = UsdGeom.Sphere.Define(house.stage, path + "/sphere")
        sphere.CreateRadiusAttr(0.0285)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
            house.materials[("porcelain", "blue", "ceramic", "sage", "cream", "screen")[i]]
        )
        house.inventory.append(
            dict(name=f"billiards_ball_{i}", path=path, category="billiard_ball", evidence="photo_048", mass_kg=0.17)
        )
    house.inventory.append(
        dict(name="billiards_table", path=body, category="pool_table", evidence="photos_048_049_057", mass_kg=300)
    )


def bathroom_details(house: BearLoopHouse) -> None:
    """Add separately mounted toilets, vessel basins, faucets, mirrors and toiletries [m]."""
    for i, room in enumerate(BATHROOMS):
        x, y = room["vanity"]
        z = room["level"] * 3.2
        path = native(
            house,
            f"bath_basin_{i}",
            "BowlFactory_974736__spawn_asset_7437077_",
            (x + 0.12, y + 0.015, z + 0.9),
            (0.40, 0.32, 0.15),
            7,
            f"bath_vanity_{i}",
            "sdf",
            anchored=True,
            evidence="photos_028_051_060",
        )
        UsdShade.MaterialBindingAPI.Apply(house.stage.GetPrimAtPath(path + "/visual")).Bind(
            house.materials["porcelain"], bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        for j, (a, b) in enumerate(
            (
                ((x + 0.12, y + 0.23, z + 0.93), (x + 0.12, y + 0.23, z + 1.22)),
                ((x + 0.12, y + 0.23, z + 1.22), (x + 0.12, y + 0.06, z + 1.22)),
            )
        ):
            house.rod(f"/World/structure/bath_faucet_{i}_{j}", a, b, 0.013, "metal")
        native(
            house,
            f"bath_toilet_{i}",
            "ToiletFactory_3350275__spawn_asset_8343411_",
            (*room["toilet"], z),
            (0.62, 0.40, 0.74),
            32,
            approximation="sdf",
            anchored=True,
            evidence="photo_060; rear lid faces the bathroom wall",
            yaw=-90,
        )
        house.asset(
            f"bath_tumbler_{i}",
            "h48:bathroom_toothbrush_tumbler_cup_0",
            (x - 0.25, y + 0.13, z + 0.9),
            (0.075, 0.075, 0.10),
            0.17,
            support=f"bath_vanity_{i}",
            kind="sdf",
        )
    native(
        house,
        "primary_bathtub",
        "BathtubFactory_9458014__spawn_asset_2435735_",
        (4.65, 2.2, 3.2),
        (1.65, 0.76, 0.58),
        85,
        approximation="sdf",
        anchored=True,
        evidence="photos_039_040",
    )


def household_variety(house: BearLoopHouse) -> None:
    """Place distinct usable possessions in appropriate furniture, with explicit support ownership."""
    # Additional closet and dressing-room possessions are inferred rather than claimed visible in photos.
    for i, (x, y, z, key, size, mass) in enumerate(
        (
            (5.8, 2.0, 3.2 + 0.8765, "h47:walk_in_closet_duffel_bag_0", (0.34, 0.22, 0.17), 1.1),
            (4.7, 1.0, 3.2 + 0.8765, "h48:closet_valet_tray_0", (0.20, 0.16, 0.035), 0.2),
            (5.8, 1.0, 3.2 + 0.8765, "h48:closet_storage_box_1", (0.30, 0.22, 0.18), 0.7),
        )
    ):
        # Existing shelf items occupy the left half; use a separate lower shelf in each cabinet.
        position = house.closet_stores[(2, 0, 1)[i]]["world"]((0, 0, 0.027))
        path = house.asset(f"closet_lower_contents_{i}", key, position, size, mass, support=f"closet_{(2, 0, 1)[i]}")
        house.stored_items.append(path)
    chest = house.storage("dressing_drawers", 4.1, 7.6, 3.2, width=0.9, height=0.82, drawer=True, yaw=-90, rows=3, material="walnut")
    house.stored(
        chest,
        [
            ("h48:closet_knit_sweater_1_1_s0_0", (0.28, 0.22, 0.065), 0.4, -0.19, 0),
            ("h48:closet_knit_sweater_1_1_s0_1", (0.24, 0.20, 0.06), 0.3, 0.17, 0),
        ],
    )
    house.asset(
        "dressing_jewelry",
        "h98:bedroom_jewelry_tray_0",
        (4.1, 7.82, 4.02),
        (0.20, 0.14, 0.035),
        0.3,
        support="dressing_drawers",
        kind="sdf",
    )
    house.asset(
        "dressing_candle",
        "h98:bedroom_scented_candle_0",
        (4.12, 7.37, 4.02),
        (0.085, 0.085, 0.12),
        0.3,
        support="dressing_drawers",
    )
    # A home office uses its own desk and chair, with a laptop, notebook and pen.
    house.asset("office_desk", "h48:bedroom_writing_desk_0", (2.25, 5.08, 3.2), (1.35, 0.65, 0.75), 25, kind="table")
    house.asset(
        "office_chair", "h48:bedroom_office_chair_0", (2.25, 5.88, 3.2), (0.60, 0.62, 1.0), 12, yaw=180, kind="chair"
    )
    for name, key, p, size, mass, yaw in (
        ("office_laptop", "h48:bedroom_slim_modern_laptop_0", (2.37, 5.12, 3.95), (0.32, 0.23, 0.18), 1.3, 5),
        ("office_notebook", "h48:bedroom_notebook_0", (1.87, 5.01, 3.95), (0.21, 0.25, 0.02), 0.3, -9),
        ("office_pen", "h98:bedroom_ballpoint_pen_0", (2.04, 4.95, 3.95), (0.012, 0.14, 0.012), 0.012, 15),
    ):
        house.asset(name, key, p, size, mass, yaw, support="office_desk")
    shelves = house.storage("library_cabinet", 0.42, 4.35, 0, width=0.95, height=1.6, yaw=90)
    books = [
        ("h98:living_room_hardcover_book_0", (0.20, 0.25, 0.035), 0.6, -0.2, 0),
        ("h98:living_room_reference_book_0", (0.19, 0.24, 0.045), 0.65, 0.16, 0),
    ]
    house.stored(shelves, books)
    for i, (key, dx, size, mass) in enumerate(
        (
            ("h98:living_room_card_game_box_0", -0.24, (0.095, 0.14, 0.035), 0.12),
            ("h98:living_room_board_game_box_1", 0.10, (0.28, 0.23, 0.065), 0.7),
        )
    ):
        p = house.asset(
            f"library_lower_contents_{i}",
            key,
            shelves["world"]((dx, 0, 0.027)),
            size,
            mass,
            yaw=90,
            support="library_cabinet",
        )
        house.stored_items.append(p)
    house.asset("living_side_table", "h48:bedroom_side_table_0", (4.65, 8.2, 0), (0.5, 0.5, 0.53), 8, kind="table")
    house.asset(
        "living_photo",
        "h98:living_room_picture_frame_0",
        (4.65, 8.23, 0.53),
        (0.18, 0.075, 0.23),
        0.35,
        support="living_side_table",
    )
    house.asset(
        "living_coaster",
        "h98:living_room_cork_drink_coaster_0",
        (2.18, 7.37, 0.43),
        (0.10, 0.10, 0.008),
        0.03,
        support="living_coffee_table",
    )
    house.asset(
        "entry_umbrella",
        "h98:hallway_folded_umbrella_0",
        (5.98, 2.4, 0.48),
        (0.085, 0.70, 0.08),
        0.5,
        support="entry_bench",
    )
    house.asset("entry_bench", "h48:hallway_entry_bench_0", (5.85, 2.35, 0), (0.45, 1.15, 0.48), 14, kind="table")
    house.asset(
        "entry_polish",
        "h48:hallway_shoe_polish_tin_0",
        (5.82, 2.65, 0.48),
        (0.075, 0.075, 0.035),
        0.08,
        support="entry_bench",
    )
    house.asset(
        "entry_sneaker",
        "h48:hallway_left_sneaker_0",
        (5.82, 2.07, 0.48),
        (0.12, 0.28, 0.11),
        0.4,
        support="entry_bench",
        yaw=12,
    )


def furnish_reference_details(house: BearLoopHouse) -> None:
    """Complete the main reference furnishing groups using explicit object placements."""
    house.replace_visual(
        "dining_table", "TableDiningFactory_5760614__spawn_asset_5974757_", (0, 0, 0.385), (1.2, 2.8, 0.77), turn=True
    )
    source_name = "TableDiningFactory_5760614__spawn_asset_5974757_"
    source_mesh = UsdGeom.Mesh(house.reference.GetPrimAtPath(house.meshes[source_name]))
    record = next(item for item in house.inventory if item["name"] == "dining_table")
    record["asset"] = source_name
    record["geometry_sha256"] = hashlib.sha256(np.asarray(source_mesh.GetPointsAttr().Get()).tobytes()).hexdigest()
    record["triangles"] = sum(count - 2 for count in source_mesh.GetFaceVertexCountsAttr().Get())
    for prim in house.stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        path = str(prim.GetPath())
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() == "UsdPreviewSurface" and "bedroom_bed_" in path:
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    leather = UsdShade.Material.Define(house.stage, "/World/materials/brown_leather")
    leather_shader = UsdShade.Shader.Define(house.stage, str(leather.GetPath()) + "/surface")
    leather_shader.CreateIdAttr("UsdPreviewSurface")
    leather_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.09, 0.035, 0.014))
    leather_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    leather_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    leather.CreateSurfaceOutput().ConnectToSource(leather_shader.ConnectableAPI(), "surface")
    for name in ("living_sofa", "living_armchair"):
        UsdShade.MaterialBindingAPI.Apply(house.stage.GetPrimAtPath("/World/objects/" + name)).Bind(
            leather, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
    house.fixture("corner_bedroom", 19.3, -0.9, 6.2, 16000)
    # The living and upper family rooms have non-burning stone fireplaces.
    fireplace(house, "living_fireplace", 0.23, 7.6, 0)
    fireplace(house, "primary_fireplace", 0.23, 7.4, 3.2)
    kitchen_contents(house)
    fridge = house.storage("refrigerator", 9.95, 9.1, 0, width=0.8, height=1.85, material="metal")
    house.stored(fridge, [("h98:bedroom_glass_tumbler_0", (0.08, 0.08, 0.11), 0.2, -0.19, 0)])
    breakfast = house.body("breakfast_table", (13.6, 10.2, 0.002), 25)
    table = UsdGeom.Cylinder.Define(house.stage, breakfast + "/top")
    table.CreateRadiusAttr(0.52)
    table.CreateHeightAttr(0.055)
    table.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.73))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(house.materials["walnut"])
    house.box(breakfast + "/pedestal", (0, 0, 0.35), (0.16, 0.16, 0.70), "walnut")
    house.box(breakfast + "/base", (0, 0, 0.045), (0.63, 0.63, 0.09), "walnut")
    for i, (x, y, yaw) in enumerate(((12.75, 10.2, -90), (14.45, 10.25, 90))):
        house.asset(
            f"breakfast_chair_{i}",
            "h48:closet_dressing_chair_0",
            (x, y, 0),
            (0.48, 0.50, 0.9),
            5,
            yaw,
            kind="chair",
            evidence="photo_018",
        )
    bathroom_details(house)
    # Move the earlier generic gaming bench to the wall, freeing the photographed pool-table group.
    house.pose("recreation_bench", (19.7, 9.6, 3.202), yaw=-26)
    house.pose("recreation_pouf", (21.25, 6.3, 3.202))
    pool_table(house)
    household_variety(house)
    shower_and_display_details(house)
    east_sleeping_rooms(house)
    for prim in house.stage.Traverse():
        if prim.GetName() == "top" and any(
            word in str(prim.GetPath()) for word in ("kitchen_", "wet_bar_", "bath_vanity_")
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(house.materials["granite"])


def shower_and_display_details(house: BearLoopHouse) -> None:
    """Complete bathroom, wall displays and household storage with appropriate mounts [m]."""
    x, y, z = 6.1, 2.35, 3.2
    root = house.body("primary_shower", (x, y, z), 45, anchored=True)
    for name, center, size, material in (
        ("base", (0, 0, 0.025), (0.92, 0.98, 0.05), "porcelain"),
        ("left", (-0.46, 0, 1.02), (0.014, 0.98, 1.95), "glass"),
        ("right", (0.46, 0, 1.02), (0.014, 0.98, 1.95), "glass"),
        ("back", (0, -0.49, 1.02), (0.92, 0.014, 1.95), "glass"),
    ):
        house.box(root + "/" + name, center, size, material)
    door = house.body("primary_shower_door", (x, y + 0.51, z + 1.025), 12)
    house.box(door + "/glass", (0, 0, 0), (0.88, 0.012, 1.95), "glass")
    for side in (-1, 1):
        house.box(
            door + ("/frame_left" if side < 0 else "/frame_right"), (side * 0.445, 0, 0), (0.025, 0.025, 1.96), "metal"
        )
    house.rod(door + "/handle", (-0.28, 0.055, -0.05), (-0.28, 0.055, 0.15), 0.012, "metal")
    for i, height in enumerate((-0.05, 0.15)):
        house.rod(door + f"/handle_mount_{i}", (-0.28, 0, height), (-0.28, 0.055, height), 0.014, "metal")
    # Place the hinge outside the frame's front corner so its sweep clears the fixed side panel.
    house.joint("primary_shower_hinge", door, (x + 0.46, y + 0.535, z + 1.025), (0.46, 0.025, 0))
    hinge = UsdPhysics.Joint(house.stage.GetPrimAtPath("/World/joints/primary_shower_hinge"))
    flip = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat())
    hinge.CreateLocalRot0Attr(flip)
    hinge.CreateLocalRot1Attr(flip)
    house.asset(
        "shower_conditioner",
        "h48:bathroom_shower_bottle_1",
        (x + 0.26, y + 0.26, z + 0.052),
        (0.075, 0.065, 0.19),
        0.35,
        support="primary_shower",
    )
    house.rod("/World/structure/shower_riser", (x, y - 0.43, z + 0.9), (x, y - 0.43, z + 1.85), 0.014, "metal")
    house.rod("/World/structure/shower_arm", (x, y - 0.43, z + 1.85), (x, y - 0.15, z + 1.85), 0.014, "metal")
    # Flush wall-mounted frames have independently authored fixed joints, never embedded dynamic bodies.
    for name, key, position, size, yaw in (
        ("living_art", "h98:living_room_painting_0", (0.12, 3.3, 1.2), (1.0, 0.045, 0.8), 90),
        ("bar_art", "h47:hallway_print_0", (6.56, 1.4, 1.3), (0.75, 0.045, 0.95), -90),
        ("primary_art", "h47:bedroom_canvas_artwork_0", (0.12, 5.7, 4.4), (0.90, 0.045, 0.75), 90),
        ("south_suite_art", "h98:bedroom_art_print_0", (0.12, 4.05, 4.5), (0.045, 0.80, 0.65), 180),
        ("office_pinboard", "h98:bedroom_pinboard_0", (2.25, 4.76, 4.35), (0.80, 0.035, 0.65), 0),
    ):
        path = house.body(name, position, 2, anchored=True, yaw=yaw)
        record = house.library.visual(key, path + "/visual", size)
        house.box(path + "/shape", (0, 0, size[2] / 2), size, "cream").GetAttribute("visibility").Set("invisible")
        record.update(name=name, path=path, evidence="photo-guided wall decor; artwork approximation", mass_kg=2)
        house.inventory.append(record)
    for i, (x, y, z) in enumerate(((5.05, 4.2, 3.2), (13.8, 4.45, 3.2))):
        house.box(f"/World/structure/vanity_mirror_frame_{i}", (x, y + 0.34, z + 1.55), (0.80, 0.045, 0.78), "walnut")
        # Metallic low-roughness plane reflects the actual room rather than a backdrop image.
        if "mirror" not in house.materials:
            material = UsdShade.Material.Define(house.stage, "/World/materials/mirror")
            shader = UsdShade.Shader.Define(house.stage, "/World/materials/mirror/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.92))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.02)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            house.materials["mirror"] = material
        house.box(f"/World/structure/vanity_mirror_{i}", (x, y + 0.312, z + 1.55), (0.73, 0.008, 0.71), "mirror")
    # Extra possessions are spread between lower shelves and surfaces rather than repeated on every table.
    unit = house.storage("recreation_game_storage", 15.65, 6.4, 3.2, width=0.95, height=1.3)
    house.stored(
        unit,
        [
            ("h98:gaming_room_charging_dock_0", (0.14, 0.11, 0.045), 0.2, -0.22, 0),
            ("h98:gaming_room_gaming_accessory_box_0", (0.24, 0.19, 0.09), 0.4, 0.17, 0),
        ],
    )
    house.asset(
        "recreation_television",
        "h98:gaming_room_tv_0",
        (15.65, 6.4, 4.5),
        (0.86, 0.14, 0.52),
        8,
        support="recreation_game_storage",
    )
    house.asset(
        "recreation_blue_game",
        "h98:gaming_room_video_game_case_blue_1",
        (15.97, 6.2, 4.5),
        (0.14, 0.19, 0.016),
        0.12,
        support="recreation_game_storage",
        yaw=13,
    )
    house.asset(
        "upper_succulent",
        "h98:living_room_potted_succulent_plant_0",
        (13.48, 6.84, 3.75),
        (0.11, 0.11, 0.20),
        0.35,
        support="upper_side_table",
    )
    house.asset(
        "entry_succulent",
        "h47:hallway_potted_succulent_plant_0",
        (4.30, 0.47, 0.9),
        (0.12, 0.12, 0.18),
        0.4,
        support="entry_console",
    )


def east_sleeping_rooms(house: BearLoopHouse) -> None:
    """Add the final east bedroom and the reference's timber bunk-bed sleeping area [m]."""
    x, y, z, yaw = 19.09, -1.525, 3.2, 154
    body = house.body("corner_bed_frame", (x, y, z + 0.002), 65, yaw=yaw)
    source = "BedFactory_1349120__spawn_asset_3701424_"
    house.mesh(source, body + "/visual", (0, 0, 0.475), (1.6, 2.1, 0.95))
    panels = [
        ("base", (0, 0, 0.30), (1.5, 1.9, 0.15)),
        ("head", (0, 1.005, 0.475), (1.6, 0.07, 0.95)),
        ("foot", (0, -1.005, 0.225), (1.6, 0.07, 0.45)),
    ]
    panels += [
        (f"leg_{i}", (dx, dy, 0.1125), (0.10, 0.10, 0.225))
        for i, (dx, dy) in enumerate(((-0.70, -0.92), (0.70, -0.92), (-0.70, 0.92), (0.70, 0.92)))
    ]
    for name, center, size in panels:
        collider = house.box(body + "/" + name, center, size, "walnut")
        UsdGeom.Imageable(collider).MakeInvisible()
    house.inventory.append(
        dict(name="corner_bed_frame", path=body, source=source, category="bed", evidence="photo_059", mass_kg=65)
    )
    mattress = "MattressFactory_1349120__spawn_asset_3701424_"
    native(house, "corner_mattress", mattress, (x, y, z + 0.377), (1.47, 1.84, 0.20), 22, "corner_bed_frame", yaw=yaw)
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
    for i, dx in enumerate((-0.36, 0.36)):
        p = Gf.Vec3d(x, y, z + 0.581) + rotation.TransformDir(Gf.Vec3d(dx, 0.55, 0))
        native(
            house,
            f"corner_pillow_{i}",
            "PillowFactory_1349120__spawn_asset_3701424_",
            tuple(p),
            (0.52, 0.36, 0.10),
            0.6,
            "corner_mattress",
            yaw=yaw,
        )
    stand = house.storage("corner_nightstand", 20.35, 0.4, 3.2, width=0.55, height=0.62, drawer=True, rows=2)
    house.stored(stand, [("h47:walk_in_closet_clutch_purse_0", (0.18, 0.12, 0.055), 0.3, 0, 0)])
    house.asset(
        "corner_alarm",
        "h98:bedroom_digital_alarm_clock_0",
        (20.35, 0.4, 3.82),
        (0.13, 0.07, 0.06),
        0.2,
        support="corner_nightstand",
    )
    # Two mattress bodies and pillows remain independent of the rigid timber frame.
    x, y, z, yaw = 22.0, 8.1, 3.2, 64
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
    frame = house.body("timber_bunk_frame", (x, y, z + 0.002), 85, yaw=yaw)
    for i, (dx, dy) in enumerate(((-0.50, -1), (0.50, -1), (-0.50, 1), (0.50, 1))):
        house.rod(frame + f"/post_{i}", (dx, dy, 0), (dx, dy, 2.12), 0.055, "oak")
    for level, top in enumerate((0.42, 1.42)):
        house.box(frame + f"/deck_{level}", (0, 0, top - 0.035), (1.00, 1.94, 0.07), "oak")
        native(
            house,
            f"bunk_mattress_{level}",
            "MattressFactory_2446410__spawn_asset_5971071_",
            (x, y, z + top + 0.004),
            (0.94, 1.84, 0.15),
            14,
            "timber_bunk_frame",
            yaw=yaw,
        )
        p = Gf.Vec3d(x, y, z + top + 0.158) + rotation.TransformDir(Gf.Vec3d(0, 0.59, 0))
        native(
            house,
            f"bunk_pillow_{level}",
            "PillowFactory_2446410__spawn_asset_5971071_",
            tuple(p),
            (0.51, 0.34, 0.09),
            0.5,
            f"bunk_mattress_{level}",
            yaw=yaw,
        )
    for side in (-1, 1):
        house.rod(
            frame + ("/guard_left" if side < 0 else "/guard_right"),
            (side * 0.50, -1, 1.88),
            (side * 0.50, 1, 1.88),
            0.035,
            "oak",
        )
    for i, height in enumerate((0.25, 0.55, 0.85, 1.15, 1.45)):
        house.rod(frame + f"/ladder_{i}", (-0.40, -1.055, height), (0.40, -1.055, height), 0.025, "oak")
    house.inventory.append(
        dict(name="timber_bunk_frame", path=frame, category="bunk_bed", evidence="photos_052_061", mass_kg=85)
    )
