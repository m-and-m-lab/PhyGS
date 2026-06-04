#!/usr/bin/env python3
"""Sort Infinigen-style USD/USDC stages into `objects` and `structure` scopes.

The script supports:
- explicit input file list
- directory scan (recursive or non-recursive)
- scene-root auto-detection
- deterministic reclassification of existing scope content
- conservative interactive-search sorting policy

Output is written next to each input file with `_sorted` suffix.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SUPPORTED_EXTENSIONS = {".usd", ".usdc"}
OBJECT_SCOPE_NAME = "objects"
STRUCTURE_SCOPE_NAME = "structure"

# Top-level Infinigen asset families that are not scene objects in the interactive-search sense.
INFINIGEN_NON_INTERACTIVE_FAMILIES = {
    "composition",
    "fluid",
    "fonts",
    "lighting",
    "materials",
    "placement",
    "scatters",
    "static_assets",
    "static",
    "utils",
    "weather",
}

# Known Infinigen object categories from infinigen/assets/objects plus legacy naming seen in docs.
INFINIGEN_OBJECT_CATEGORIES = {
    "appliances",
    "basket",
    "bathroom",
    "cactus",
    "ceiling_classic_lamp",
    "chairs",
    "cloud",
    "clothes",
    "corals",
    "creatures",
    "decor",
    "deformed_trees",
    "doors",
    "elements",
    "fruits",
    "grassland",
    "lamp",
    "leaves",
    "mollusk",
    "monocot",
    "mushroom",
    "organizer",
    "particles",
    "plate_rack",
    "rocks",
    "seating",
    "shelves",
    "small_plants",
    "staircases",
    "table_decorations",
    "tables",
    "tableware",
    "trees",
    "tropic_plants",
    "underwater",
    "wall_decorations",
    "warehouses",
    "windows",
}

# Built-in manifest: factory stem -> category.
# Derived from interactive-search/scripts/docs/count_objects.py (factories_category).
INFINIGEN_FACTORY_STEM_TO_CATEGORY = {
    "agave_monocot": "monocot",
    "altocumulus": "cloud",
    "ant_swarm": "creatures",
    "aquarium_tank": "decor",
    "arm_chair": "seating",
    "auger": "mollusk",
    "auger_base": "mollusk",
    "balloon": "wall_decorations",
    "banana_monocot": "monocot",
    "bar_chair": "seating",
    "basket_base": "organizer",
    "bathroom_sink": "bathroom",
    "bathtub": "bathroom",
    "bed": "seating",
    "bed_frame": "seating",
    "beetle": "creatures",
    "beverage_fridge": "appliances",
    "bird": "creatures",
    "blanket": "clothes",
    "blender_rock": "rocks",
    "book": "table_decorations",
    "book_column": "table_decorations",
    "book_stack": "table_decorations",
    "bottle": "tableware",
    "boulder": "rocks",
    "bowl": "tableware",
    "brain_base_coral": "corals",
    "brain_coral": "corals",
    "bush": "trees",
    "bush_base_coral": "corals",
    "bush_coral": "corals",
    "cabinet_door_base": "shelves",
    "cactus": "cactus",
    "can": "tableware",
    "cantilever_staircase": "elements",
    "carnivore": "creatures",
    "cauliflower_base_coral": "corals",
    "cauliflower_coral": "corals",
    "ceiling_classic_lamp": "lamp",
    "ceiling_light": "lamp",
    "cell_shelf": "shelves",
    "chair": "seating",
    "chopsticks": "tableware",
    "clam": "mollusk",
    "clam_base": "mollusk",
    "cloud": "cloud",
    "coconut_tree": "tropic_plants",
    "coffee_table": "tables",
    "columnar_base_cactus": "cactus",
    "columnar_cactus": "cactus",
    "conch": "mollusk",
    "conch_base": "mollusk",
    "coral": "corals",
    "countertop": "shelves",
    "crab": "creatures",
    "crustacean": "creatures",
    "cumulonimbus": "cloud",
    "cumulus": "cloud",
    "cup": "tableware",
    "curved_staircase": "elements",
    "dandelion": "grassland",
    "dandelion_seed": "grassland",
    "deformed_tree": "deformed_trees",
    "desk_lamp": "lamp",
    "diff_growth_base_coral": "corals",
    "dishwasher": "appliances",
    "door_casing": "elements",
    "dragonfly": "creatures",
    "dust_mote": "particles",
    "elkhorn_base_coral": "corals",
    "elkhorn_coral": "corals",
    "fallen_tree": "deformed_trees",
    "fan_base_coral": "corals",
    "fan_coral": "corals",
    "fern": "small_plants",
    "fish": "creatures",
    "fish_school": "creatures",
    "floor_lamp": "lamp",
    "flower": "grassland",
    "flower_plant": "grassland",
    "flying_bird": "creatures",
    "food_bag": "tableware",
    "food_box": "tableware",
    "fork": "tableware",
    "frog": "creatures",
    "fruit_container": "tableware",
    "glass_panel_door": "elements",
    "globular_base_cactus": "cactus",
    "globular_cactus": "cactus",
    "glowing_rocks": "rocks",
    "grass_tuft": "grassland",
    "grasses_monocot": "monocot",
    "hardware": "bathroom",
    "herbivore": "creatures",
    "hollow_tree": "deformed_trees",
    "honeycomb_base_coral": "corals",
    "honeycomb_coral": "corals",
    "hook_base": "organizer",
    "jar": "tableware",
    "jellyfish": "creatures",
    "kalidium_base_cactus": "cactus",
    "kalidium_cactus": "cactus",
    "kelp_monocot": "monocot",
    "kitchen_cabinet": "shelves",
    "kitchen_island": "shelves",
    "kitchen_space": "shelves",
    "knife": "tableware",
    "l_shaped_staircase": "elements",
    "lamp": "lamp",
    "large_plant_container": "tableware",
    "large_shelf": "shelves",
    "leaf": "leaves",
    "leaf_banana_tree": "tropic_plants",
    "leaf_palm_plant": "tropic_plants",
    "leaf_palm_tree": "tropic_plants",
    "leather_base_coral": "corals",
    "leather_coral": "corals",
    "lichen": "particles",
    "lid": "tableware",
    "lite_door": "elements",
    "lizard": "creatures",
    "lobster": "creatures",
    "louver_door": "elements",
    "maize_monocot": "monocot",
    "mattress": "seating",
    "microwave": "appliances",
    "mirror": "wall_decorations",
    "mollusk": "mollusk",
    "monitor": "appliances",
    "monocot": "monocot",
    "moss": "particles",
    "mushroom": "mushroom",
    "mushroom_growth": "mushroom",
    "mussel": "mollusk",
    "mussel_base": "mollusk",
    "nature_shelf_trinkets": "elements",
    "nautilus": "mollusk",
    "nautilus_base": "mollusk",
    "office_chair": "seating",
    "oven": "appliances",
    "pallet": "elements",
    "palm_tree": "tropic_plants",
    "pan": "tableware",
    "panel_door": "elements",
    "pants": "clothes",
    "pillar": "elements",
    "pillow": "seating",
    "pine_needle": "particles",
    "pinecone": "monocot",
    "plant_banana_tree": "tropic_plants",
    "plant_container": "tableware",
    "plate": "tableware",
    "plate_on_rack_base": "organizer",
    "plate_rack_base": "organizer",
    "pot": "tableware",
    "pricky_pear_base_cactus": "cactus",
    "pricky_pear_cactus": "cactus",
    "rack": "elements",
    "raindrop": "particles",
    "range_hood": "wall_decorations",
    "reaction_diffusion_base_coral": "corals",
    "rotten_tree": "deformed_trees",
    "rug": "elements",
    "scallop": "mollusk",
    "scallop_base": "mollusk",
    "seaweed": "underwater",
    "shell_base": "mollusk",
    "shirt": "clothes",
    "side_table": "tables",
    "sidetable_desk": "shelves",
    "simple_bookcase": "shelves",
    "simple_desk": "shelves",
    "single_cabinet": "shelves",
    "single_cabinet_articulated": "shelves",
    "sink": "table_decorations",
    "snail_base": "mollusk",
    "snake": "creatures",
    "snake_plant": "small_plants",
    "snowflake": "particles",
    "sofa": "seating",
    "spatula": "tableware",
    "spatula_on_hook_base": "organizer",
    "spider_plant": "small_plants",
    "spiny_lobster": "creatures",
    "spiral_staircase": "elements",
    "spoon": "tableware",
    "standing_sink": "bathroom",
    "star_base_coral": "corals",
    "star_coral": "corals",
    "straight_staircase": "elements",
    "stratocumulus": "cloud",
    "succulent": "small_plants",
    "table_base_coral": "corals",
    "table_cocktail": "tables",
    "table_coral": "corals",
    "table_dining": "tables",
    "table_top": "tables",
    "tap": "table_decorations",
    "taro_monocot": "monocot",
    "toilet": "bathroom",
    "towel": "seating",
    "tree": "trees",
    "tree_base_coral": "corals",
    "tree_flower": "trees",
    "triangle_shelf": "shelves",
    "truncated_tree": "deformed_trees",
    "tube_base_coral": "corals",
    "tube_coral": "corals",
    "tussock_monocot": "monocot",
    "tv": "appliances",
    "tv_stand": "shelves",
    "twig_base_coral": "corals",
    "twig_coral": "corals",
    "u_shaped_staircase": "elements",
    "urchin": "underwater",
    "vase": "table_decorations",
    "veratrum_monocot": "monocot",
    "volute": "mollusk",
    "volute_base": "mollusk",
    "wall_art": "wall_decorations",
    "wall_shelf": "wall_decorations",
    "wheat_ear_monocot": "monocot",
    "wheat_monocot": "monocot",
    "window": "windows",
    "wineglass": "tableware",
}

# Built-in manifest: common stems from infinigen/assets/sim_objects family.
INFINIGEN_SIM_OBJECT_STEMS = {
    "box",
    "cabinet",
    "dishwasher",
    "door",
    "door_handle",
    "drawer",
    "faucet",
    "lamp",
    "microwave",
    "oven",
    "pepper_grinder",
    "plier",
    "refrigerator",
    "soap_dispenser",
    "stovetop",
    "toaster",
    "trash",
    "window",
}

# Category defaults for conservative interactive-search splitting.
MOVABLE_OBJECT_CATEGORIES = {
    "appliances",
    "basket",
    "clothes",
    "decor",
    "fruits",
    "lamp",
    "organizer",
    "plate_rack",
    "shelves",
    "table_decorations",
    "tables",
    "tableware",
    "wall_decorations",
    "warehouses",
}

STRUCTURE_OBJECT_CATEGORIES = {
    "bathroom",
    "cactus",
    "ceiling_classic_lamp",
    "chairs",
    "cloud",
    "corals",
    "creatures",
    "deformed_trees",
    "doors",
    "elements",
    "grassland",
    "leaves",
    "mollusk",
    "monocot",
    "mushroom",
    "particles",
    "rocks",
    "seating",
    "small_plants",
    "staircases",
    "trees",
    "tropic_plants",
    "underwater",
    "windows",
}

# Explicit per-stem conservative overrides even if category is usually movable.
FORCE_STRUCTURE_FACTORY_STEMS = {
    "arm_chair",
    "bar_chair",
    "bed",
    "bed_frame",
    "blanket",
    "ceiling_classic_lamp",
    "ceiling_light",
    "chair",
    "large_plant_container",
    "mattress",
    "mirror",
    "office_chair",
    "plant_container",
    "pillow",
    "point_lamp",
    "sink",
    "sofa",
    "tap",
    "tower",
    "wall_art",
}

# Explicit per-stem overrides that should remain in objects.
FORCE_OBJECT_FACTORY_STEMS = {
    "nature_shelf_trinkets",
    "tv",
}

# Keywords that must be evaluated before factory/category fallback.
EARLY_FORCE_STRUCTURE_KEYWORDS = {
    "point_lamp",
    "pointlamp",
}

# Explicit keyword overrides.
STRUCTURE_KEYWORDS_BASE = {
    "architecture",
    "bathroom",
    "bed",
    "bedframe",
    "bed_frame",
    "blanket",
    "building",
    "ceiling",
    "chair",
    "column",
    "corridor",
    "door",
    "doorframe",
    "door_frame",
    "elevator",
    "entryway",
    "exterior",
    "facade",
    "floor",
    "foundation",
    "hallway",
    "house",
    "large_plant",
    "layout",
    "mattress",
    "mirror",
    "pillow",
    "pillar",
    "plant",
    "point_lamp",
    "pointlamp",
    "railing",
    "roof",
    "room",
    "sink",
    "sofa",
    "stair",
    "staircase",
    "support",
    "tap",
    "terrain",
    "tile",
    "tower",
    "trim",
    "wall",
    "wall_art",
    "window",
}

OBJECT_KEYWORDS_BASE = {
    "appliance",
    "armchair",
    "arm_chair",
    "bar_chair",
    "basket",
    "book",
    "bookcase",
    "bottle",
    "bowl",
    "cabinet",
    "can",
    "chair",
    "chopsticks",
    "coffee_table",
    "container",
    "countertop",
    "cup",
    "decor",
    "desk",
    "dish",
    "dishwasher",
    "drawer",
    "fork",
    "fridge",
    "fruit",
    "glass",
    "jar",
    "knife",
    "lamp",
    "laptop",
    "lid",
    "microwave",
    "monitor",
    "mug",
    "oven",
    "pan",
    "plant_container",
    "plate",
    "plier",
    "pot",
    "rack",
    "refrigerator",
    "shelf",
    "sidetable",
    "side_table",
    "sink",
    "soap_dispenser",
    "sofa",
    "spatula",
    "spoon",
    "stool",
    "table",
    "tableware",
    "toaster",
    "tool",
    "trash",
    "trinket",
    "tv",
    "tvstand",
    "tv_stand",
    "vase",
    "wineglass",
}

# Skip known non-scene infrastructure branches.
SKIP_TOKENS = {
    "camera",
    "cameras",
    "dome_light",
    "light",
    "lights",
    "look",
    "looks",
    "material",
    "materials",
    "physics",
    "physx",
    "render",
    "shader",
    "texture",
    "textures",
}


@dataclass
class CandidatePrim:
    src_path: str
    prim_name: str


@dataclass
class FileStats:
    input_path: Path
    output_path: Path
    scene_root: str
    total_candidates: int = 0
    moved: int = 0
    already_in_place: int = 0
    object_count: int = 0
    structure_count: int = 0
    skipped_infra: int = 0
    unknown_to_structure: int = 0
    renamed_on_collision: int = 0


@dataclass
class ClassifierConfig:
    object_keywords: set[str]
    structure_keywords: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort Infinigen-style USD/USDC prims into /objects and /structure scopes."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=[],
        help="USD/USDC file path(s) to process.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Directory to scan for USD/USDC files.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan input directory recursively (default: True).",
    )
    parser.add_argument(
        "--scene-root",
        type=str,
        default="",
        help="Optional explicit scene root prim path (e.g., /World/Scene).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Classify and print actions without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-prim classification and move details.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing *_sorted.usd/usdc outputs.",
    )
    parser.add_argument(
        "--extra-object-keywords",
        type=str,
        nargs="*",
        default=[],
        help="Extra object keywords (space/comma-separated).",
    )
    parser.add_argument(
        "--extra-structure-keywords",
        type=str,
        nargs="*",
        default=[],
        help="Extra structure keywords (space/comma-separated).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        default=False,
        help="Run synthetic classifier checks and exit.",
    )
    return parser.parse_args()


def _normalize_blob(value: str) -> str:
    # Split acronym-to-word boundaries, e.g. TVFactory -> TV_Factory.
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if not value:
        return value

    # Collapse acronym-style runs split by underscores, e.g. t_v -> tv.
    tokens = value.split("_")
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if len(token) == 1 and token.isalpha():
            j = i
            letters: list[str] = []
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                letters.append(tokens[j])
                j += 1
            if len(letters) >= 2:
                merged.append("".join(letters))
                i = j
                continue
        merged.append(token)
        i += 1

    return "_".join(merged)


def _split_keywords(values: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        for token in re.split(r"[\s,]+", value.strip()):
            normalized = _normalize_blob(token)
            if normalized:
                out.add(normalized)
    return out


def _build_classifier_config(args: argparse.Namespace) -> ClassifierConfig:
    object_keywords = set(OBJECT_KEYWORDS_BASE)
    structure_keywords = set(STRUCTURE_KEYWORDS_BASE)
    object_keywords.update(_split_keywords(args.extra_object_keywords))
    structure_keywords.update(_split_keywords(args.extra_structure_keywords))

    # Keep category labels as keywords too.
    object_keywords.update(MOVABLE_OBJECT_CATEGORIES)
    structure_keywords.update(STRUCTURE_OBJECT_CATEGORIES)

    # Ensure precedence stays conservative.
    object_keywords -= structure_keywords

    return ClassifierConfig(
        object_keywords=object_keywords,
        structure_keywords=structure_keywords,
    )


def _require_pxr_bindings() -> None:
    """Fail early with an actionable message when pxr is unavailable."""
    try:
        from pxr import Usd  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing USD Python bindings (`pxr`). "
            "Run this script in an environment with OpenUSD installed.\n"
            "Recommended for this repo:\n"
            "  cd /workspace/isaaclab\n"
            "  ./isaaclab.sh -p -m pip install usd-core\n"
            "  ./isaaclab.sh -p scripts/interactive-search/scripts/helpers/sort_infinigen_usd_scopes.py ..."
        ) from exc


def _contains_phrase(padded_blob: str, phrase: str) -> bool:
    return f"_{phrase}_" in padded_blob


def _pick_keyword_match(padded_blob: str, keywords: set[str]) -> str | None:
    for keyword in sorted(keywords, key=len, reverse=True):
        if _contains_phrase(padded_blob, keyword):
            return keyword
    return None


def _match_factory_stem_to_category(padded_blob: str) -> tuple[str, str] | None:
    matches: list[tuple[str, str]] = []
    for stem, category in INFINIGEN_FACTORY_STEM_TO_CATEGORY.items():
        if _contains_phrase(padded_blob, stem):
            matches.append((stem, category))
    if not matches:
        return None
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    return matches[0]


def _match_sim_object_stem(padded_blob: str) -> str | None:
    matches = [stem for stem in INFINIGEN_SIM_OBJECT_STEMS if _contains_phrase(padded_blob, stem)]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


def classify_prim(prim_name: str, prim_path: str, cfg: ClassifierConfig) -> tuple[str, str, bool]:
    """Classify prim into objects/structure.

    Returns:
        (target_scope_name, reason, used_unknown_fallback)
    """
    normalized_blob = _normalize_blob(f"{prim_name}_{prim_path}")
    padded_blob = f"_{normalized_blob}_"

    early_structure_kw = _pick_keyword_match(padded_blob, EARLY_FORCE_STRUCTURE_KEYWORDS)
    if early_structure_kw is not None:
        return STRUCTURE_SCOPE_NAME, f"early-structure-keyword:{early_structure_kw}", False

    if _pick_keyword_match(padded_blob, INFINIGEN_NON_INTERACTIVE_FAMILIES) is not None:
        return STRUCTURE_SCOPE_NAME, "non-interactive-family", False

    stem_category = _match_factory_stem_to_category(padded_blob)
    if stem_category is not None:
        stem, category = stem_category
        if stem in FORCE_OBJECT_FACTORY_STEMS:
            return OBJECT_SCOPE_NAME, f"forced-object-stem:{stem}", False
        if stem in FORCE_STRUCTURE_FACTORY_STEMS:
            return STRUCTURE_SCOPE_NAME, f"forced-structure-stem:{stem}", False
        if category in MOVABLE_OBJECT_CATEGORIES:
            return OBJECT_SCOPE_NAME, f"factory-category:{category}:{stem}", False
        return STRUCTURE_SCOPE_NAME, f"factory-category:{category}:{stem}", False

    sim_stem = _match_sim_object_stem(padded_blob)
    if sim_stem is not None:
        if sim_stem in {"door", "window"}:
            return STRUCTURE_SCOPE_NAME, f"sim-object:{sim_stem}", False
        return OBJECT_SCOPE_NAME, f"sim-object:{sim_stem}", False

    structure_kw = _pick_keyword_match(padded_blob, cfg.structure_keywords)
    if structure_kw is not None:
        return STRUCTURE_SCOPE_NAME, f"structure-keyword:{structure_kw}", False

    object_kw = _pick_keyword_match(padded_blob, cfg.object_keywords)
    if object_kw is not None:
        return OBJECT_SCOPE_NAME, f"object-keyword:{object_kw}", False

    matched_category = _pick_keyword_match(padded_blob, INFINIGEN_OBJECT_CATEGORIES)
    if matched_category is not None:
        if matched_category in MOVABLE_OBJECT_CATEGORIES:
            return OBJECT_SCOPE_NAME, f"category:{matched_category}", False
        return STRUCTURE_SCOPE_NAME, f"category:{matched_category}", False

    return STRUCTURE_SCOPE_NAME, "unknown->structure", True


def _is_supported_usd_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _collect_input_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []

    for raw in args.inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected file but got directory in --inputs: {path}")
        if not _is_supported_usd_file(path):
            raise ValueError(f"Unsupported file extension (expected .usd/.usdc): {path}")
        files.append(path.resolve())

    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"--input-dir is not a valid directory: {input_dir}")
        iterator: Iterable[Path]
        if args.recursive:
            iterator = input_dir.rglob("*")
        else:
            iterator = input_dir.glob("*")
        for path in sorted(iterator):
            if path.is_file() and _is_supported_usd_file(path):
                files.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for file_path in files:
        if file_path not in seen:
            seen.add(file_path)
            deduped.append(file_path)

    if not deduped:
        raise ValueError("No USD/USDC files found. Provide --inputs and/or --input-dir.")

    return deduped


def _build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sorted{input_path.suffix}")


def _resolve_scene_root(stage: Any, override_scene_root: str) -> str:
    if override_scene_root:
        prim = stage.GetPrimAtPath(override_scene_root)
        if not prim or not prim.IsValid():
            raise ValueError(f"--scene-root does not exist on stage: {override_scene_root}")
        return prim.GetPath().pathString

    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim.GetPath().pathString

    world_prim = stage.GetPrimAtPath("/World")
    if world_prim and world_prim.IsValid():
        return "/World"

    top_level = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim and prim.IsValid()]
    if len(top_level) == 1:
        return top_level[0].GetPath().pathString

    candidates = ", ".join(prim.GetPath().pathString for prim in top_level)
    raise RuntimeError(
        "Unable to auto-detect scene root. "
        f"Found multiple top-level prims: [{candidates}]. "
        "Pass --scene-root explicitly."
    )


def _is_infrastructure_prim(prim_name: str, prim_path: str) -> bool:
    normalized_blob = _normalize_blob(f"{prim_name}_{prim_path}")
    padded_blob = f"_{normalized_blob}_"
    return _pick_keyword_match(padded_blob, SKIP_TOKENS) is not None


def _ensure_scope(stage: Any, scope_path: str) -> Any:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(scope_path)
    if prim and prim.IsValid():
        return prim
    scope = UsdGeom.Scope.Define(stage, scope_path)
    return scope.GetPrim()


def _collect_candidates(stage: Any, scene_root: str, stats: FileStats) -> list[CandidatePrim]:
    root_prim = stage.GetPrimAtPath(scene_root)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Scene root prim is invalid: {scene_root}")

    objects_scope_path = f"{scene_root}/{OBJECT_SCOPE_NAME}"
    structure_scope_path = f"{scene_root}/{STRUCTURE_SCOPE_NAME}"

    candidates_by_path: dict[str, CandidatePrim] = {}

    def add_children(parent_prim: Any) -> None:
        if not parent_prim or not parent_prim.IsValid():
            return
        for child in parent_prim.GetChildren():
            child_path = child.GetPath().pathString
            if child_path in {objects_scope_path, structure_scope_path}:
                continue
            prim_name = child.GetName()
            if _is_infrastructure_prim(prim_name, child_path):
                stats.skipped_infra += 1
                continue
            candidates_by_path[child_path] = CandidatePrim(src_path=child_path, prim_name=prim_name)

    add_children(root_prim)
    add_children(stage.GetPrimAtPath(objects_scope_path))
    add_children(stage.GetPrimAtPath(structure_scope_path))

    return [candidates_by_path[path] for path in sorted(candidates_by_path.keys())]


def _choose_unique_name(name: str, reserved: set[str]) -> tuple[str, bool]:
    if name not in reserved:
        return name, False
    index = 1
    while True:
        candidate = f"{name}_{index}"
        if candidate not in reserved:
            return candidate, True
        index += 1


def _move_prim(stage: Any, src_path: str, dst_path: str) -> tuple[bool, str]:
    """Move prim path using NamespaceEditor if available, otherwise Sdf.BatchNamespaceEdit."""
    if src_path == dst_path:
        return True, "no-op"

    from pxr import Sdf, Usd

    if hasattr(Usd, "NamespaceEditor"):
        try:
            editor = Usd.NamespaceEditor(stage)
            moved = False
            if hasattr(editor, "MovePrimAtPath"):
                moved = bool(editor.MovePrimAtPath(src_path, dst_path))
            elif hasattr(editor, "MovePrim"):
                moved = bool(editor.MovePrim(src_path, dst_path))
            if moved:
                if hasattr(editor, "ApplyEdits"):
                    result = editor.ApplyEdits()
                    if result is None or bool(result):
                        return True, "namespace-editor"
                elif hasattr(editor, "Apply"):
                    result = editor.Apply()
                    if result is None or bool(result):
                        return True, "namespace-editor"
        except Exception:
            # Fall back below.
            pass

    edit = Sdf.BatchNamespaceEdit()
    edit.Add(Sdf.Path(src_path), Sdf.Path(dst_path))
    applied = stage.GetRootLayer().Apply(edit)
    if applied:
        return True, "sdf-batch-edit"

    return False, "failed"


def _process_file(input_path: Path, args: argparse.Namespace, cfg: ClassifierConfig) -> FileStats:
    from pxr import Sdf, Usd

    output_path = _build_output_path(input_path)
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite to regenerate."
        )

    stage = Usd.Stage.Open(str(input_path))
    if not stage:
        raise RuntimeError(f"Failed to open stage: {input_path}")

    scene_root = _resolve_scene_root(stage, args.scene_root)
    stats = FileStats(input_path=input_path, output_path=output_path, scene_root=scene_root)

    objects_scope_path = f"{scene_root}/{OBJECT_SCOPE_NAME}"
    structure_scope_path = f"{scene_root}/{STRUCTURE_SCOPE_NAME}"

    _ensure_scope(stage, objects_scope_path)
    _ensure_scope(stage, structure_scope_path)

    candidates = _collect_candidates(stage, scene_root, stats)
    stats.total_candidates = len(candidates)
    candidate_paths = {candidate.src_path for candidate in candidates}

    # Reserve names per target scope to avoid path collisions.
    reserved_names = {OBJECT_SCOPE_NAME: set(), STRUCTURE_SCOPE_NAME: set()}

    for scope_name, scope_path in (
        (OBJECT_SCOPE_NAME, objects_scope_path),
        (STRUCTURE_SCOPE_NAME, structure_scope_path),
    ):
        scope_prim = stage.GetPrimAtPath(scope_path)
        if not scope_prim or not scope_prim.IsValid():
            continue
        for child in scope_prim.GetChildren():
            child_path = child.GetPath().pathString
            if child_path in candidate_paths:
                continue
            reserved_names[scope_name].add(child.GetName())

    assignments: list[tuple[CandidatePrim, str, str, bool]] = []

    for candidate in candidates:
        target_scope, reason, used_unknown = classify_prim(
            prim_name=candidate.prim_name,
            prim_path=candidate.src_path,
            cfg=cfg,
        )

        if target_scope == OBJECT_SCOPE_NAME:
            stats.object_count += 1
            target_scope_path = objects_scope_path
        else:
            stats.structure_count += 1
            target_scope_path = structure_scope_path
            if used_unknown:
                stats.unknown_to_structure += 1

        chosen_name, renamed = _choose_unique_name(candidate.prim_name, reserved_names[target_scope])
        reserved_names[target_scope].add(chosen_name)

        dst_path = str(Sdf.Path(target_scope_path).AppendChild(chosen_name))
        assignments.append((candidate, dst_path, reason, renamed))

    for candidate, dst_path, reason, renamed in assignments:
        src_path = candidate.src_path
        if src_path == dst_path:
            stats.already_in_place += 1
            if args.verbose:
                print(f"[KEEP] {src_path} ({reason})")
            continue

        if renamed:
            stats.renamed_on_collision += 1

        if args.dry_run:
            if args.verbose:
                print(f"[DRY] {src_path} -> {dst_path} ({reason})")
            stats.moved += 1
            continue

        ok, method = _move_prim(stage, src_path, dst_path)
        if not ok:
            raise RuntimeError(f"Failed to move prim: {src_path} -> {dst_path}")
        stats.moved += 1
        if args.verbose:
            print(f"[MOVE:{method}] {src_path} -> {dst_path} ({reason})")

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported = stage.GetRootLayer().Export(str(output_path))
        if not exported:
            raise RuntimeError(f"Failed to export stage to: {output_path}")

    return stats


def _run_self_test(cfg: ClassifierConfig) -> int:
    tests = [
        ("BookFactory_000", "/World/Scene/BookFactory_000", OBJECT_SCOPE_NAME),
        ("TableDiningFactory_000", "/World/Scene/TableDiningFactory_000", OBJECT_SCOPE_NAME),
        ("DeskLampFactory_000", "/World/Scene/DeskLampFactory_000", OBJECT_SCOPE_NAME),
        ("FloorLampFactory_000", "/World/Scene/FloorLampFactory_000", OBJECT_SCOPE_NAME),
        ("WallShelfFactory_000", "/World/Scene/WallShelfFactory_000", OBJECT_SCOPE_NAME),
        ("BlanketFactory_000", "/World/Scene/BlanketFactory_000", STRUCTURE_SCOPE_NAME),
        ("ChairFactory_000", "/World/Scene/ChairFactory_000", STRUCTURE_SCOPE_NAME),
        ("MirrorFactory_000", "/World/Scene/MirrorFactory_000", STRUCTURE_SCOPE_NAME),
        ("LargePlantContainerFactory_000", "/World/Scene/LargePlantContainerFactory_000", STRUCTURE_SCOPE_NAME),
        ("PlantContainerFactory_000", "/World/Scene/PlantContainerFactory_000", STRUCTURE_SCOPE_NAME),
        ("PointLamp_000", "/World/Scene/PointLamp_000", STRUCTURE_SCOPE_NAME),
        ("SofaFactory_000", "/World/Scene/SofaFactory_000", STRUCTURE_SCOPE_NAME),
        ("SinkFactory_000", "/World/Scene/SinkFactory_000", STRUCTURE_SCOPE_NAME),
        ("TapFactory_000", "/World/Scene/TapFactory_000", STRUCTURE_SCOPE_NAME),
        ("Tower_000", "/World/Scene/Tower_000", STRUCTURE_SCOPE_NAME),
        ("WallArtFactory_000", "/World/Scene/WallArtFactory_000", STRUCTURE_SCOPE_NAME),
        ("NatureShelfTrinketsFactory_000", "/World/Scene/NatureShelfTrinketsFactory_000", OBJECT_SCOPE_NAME),
        ("TVFactory_000", "/World/Scene/TVFactory_000", OBJECT_SCOPE_NAME),
        ("room_0_0_wall", "/World/Scene/room_0_0_wall", STRUCTURE_SCOPE_NAME),
        ("bed_01", "/World/Scene/bed_01", STRUCTURE_SCOPE_NAME),
        ("mattress_main", "/World/Scene/mattress_main", STRUCTURE_SCOPE_NAME),
        ("pillow_left", "/World/Scene/pillow_left", STRUCTURE_SCOPE_NAME),
        ("CeilingLightFactory_000", "/World/Scene/CeilingLightFactory_000", STRUCTURE_SCOPE_NAME),
        ("DishwasherFactory_000", "/World/Scene/DishwasherFactory_000", OBJECT_SCOPE_NAME),
        ("SpiralStaircaseFactory_000", "/World/Scene/SpiralStaircaseFactory_000", STRUCTURE_SCOPE_NAME),
    ]

    failures = 0
    for prim_name, prim_path, expected in tests:
        got, reason, _ = classify_prim(prim_name, prim_path, cfg)
        if got != expected:
            failures += 1
            print(
                f"[FAIL] {prim_name} expected={expected} got={got} reason={reason}",
                file=sys.stderr,
            )
        else:
            print(f"[PASS] {prim_name} -> {got} ({reason})")

    if failures:
        print(f"[SELF-TEST] {failures} failures", file=sys.stderr)
        return 1

    print("[SELF-TEST] all checks passed")
    return 0


def main() -> int:
    args = parse_args()
    cfg = _build_classifier_config(args)

    if args.self_test:
        return _run_self_test(cfg)

    try:
        _require_pxr_bindings()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        input_files = _collect_input_files(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    all_stats: list[FileStats] = []

    for input_file in input_files:
        try:
            stats = _process_file(input_file, args, cfg)
            all_stats.append(stats)
            print(
                "[DONE] "
                f"{stats.input_path} -> {stats.output_path} "
                f"root={stats.scene_root} candidates={stats.total_candidates} "
                f"moved={stats.moved} keep={stats.already_in_place} "
                f"objects={stats.object_count} structure={stats.structure_count} "
                f"unknown_to_structure={stats.unknown_to_structure} "
                f"renamed={stats.renamed_on_collision} skipped_infra={stats.skipped_infra}"
            )
        except Exception as exc:
            print(f"[ERROR] Failed processing {input_file}: {exc}", file=sys.stderr)
            return 1

    if not all_stats:
        print("[WARN] No files processed.")
        return 0

    total_files = len(all_stats)
    total_candidates = sum(s.total_candidates for s in all_stats)
    total_moved = sum(s.moved for s in all_stats)
    total_objects = sum(s.object_count for s in all_stats)
    total_structure = sum(s.structure_count for s in all_stats)

    print(
        "[SUMMARY] "
        f"files={total_files} candidates={total_candidates} moved={total_moved} "
        f"objects={total_objects} structure={total_structure}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
