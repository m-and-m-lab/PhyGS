# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch and verify ForageBench scene releases into the local scene cache (``phygs.scenes.fetch``).

    ./isaaclab.sh -p -m phygs.scenes.fetch list
    ./isaaclab.sh -p -m phygs.scenes.fetch fetch --scene_id bear_loop_house --scene_version 1.0.0

Releases land under ``~/.cache/phygs/scenes`` (or ``--cache_dir``), with a ``<scene_id>.json`` record
the scene registry (:mod:`phygs.episodes.scenes`) reads, so a spec says ``scene: {id: bear_loop_house}``.
"""

import argparse
import hashlib
import json
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

REPO = "mandmlabresearch/ForageBench"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def install(archive: Path, manifest: dict, cache: Path) -> Path:
    """Verify and extract a scene into a content-addressed local cache."""
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported scene schema")
    checksum = manifest["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", checksum) or digest(archive) != checksum:
        raise ValueError("Archive checksum mismatch")
    entry = relative_path(manifest["entrypoint"])
    destination = cache.resolve() / checksum
    cache.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Cache already exists: {destination}; use its scene.usdc or remove it to reinstall")
    with tempfile.TemporaryDirectory(dir=cache, prefix="extract-") as temporary:
        root = Path(temporary) / "scene"
        root.mkdir()
        with ZipFile(archive) as bundle:
            names = set()
            for member in bundle.infolist():
                name = relative_path(member.filename).as_posix()
                if name in names or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Duplicate or symlink archive member: {name}")
                names.add(name)
            if sum(member.file_size for member in bundle.infolist()) != manifest["uncompressed_bytes"]:
                raise ValueError("Archive uncompressed size mismatch")
            bundle.extractall(root)
        from pxr import UsdUtils

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root / entry))
        if unresolved:
            raise ValueError(f"Unresolved USD dependencies: {unresolved}")
        dependencies = [Path(layer.realPath) for layer in layers] + [Path(asset) for asset in assets]
        for dependency in dependencies:
            resolved = dependency.resolve(strict=True)
            if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                raise ValueError(f"Dependency outside extracted scene: {dependency}")
        if not (root / entry).is_file():
            raise FileNotFoundError(entry)
        root.rename(destination)
    return destination / entry


def fetch_config(config_path: Path, cache: Path, record_path: Path) -> Path:
    """Fetch a pinned YAML scene specification and write its local provenance record."""
    import yaml
    from huggingface_hub import hf_hub_download

    config = yaml.safe_load(config_path.read_text())
    if config["schema_version"] != 1 or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Expected schema 1 and a full immutable dataset commit SHA")
    manifest_path = hf_hub_download(
        config["repo_id"],
        str(relative_path(config["manifest"])),
        repo_type="dataset",
        revision=config["revision"],
    )
    manifest = json.loads(Path(manifest_path).read_text())
    for key in ("scene_id", "scene_version", "sha256", "physics"):
        if manifest[key] != config[key]:
            raise ValueError(f"Pinned config disagrees with manifest: {key}")
    return fetch_manifest(config, manifest, cache, record_path)


def fetch_manifest(config: dict, manifest: dict, cache: Path, record_path: Path) -> Path:
    """Download one release and record its immutable origin."""
    from huggingface_hub import hf_hub_download

    archive = Path(
        hf_hub_download(
            config["repo_id"],
            str(relative_path(manifest["archive"])),
            repo_type="dataset",
            revision=config["revision"],
        )
    )
    entry = cache.resolve() / manifest["sha256"] / relative_path(manifest["entrypoint"])
    if (cache.resolve() / manifest["sha256"]).exists():
        # Verify every extracted byte against the pinned archive before reusing the cache.
        if digest(archive) != manifest["sha256"]:
            raise ValueError("Archive checksum mismatch")
        with ZipFile(archive) as bundle:
            root = cache.resolve() / manifest["sha256"]
            for member in bundle.infolist():
                if not member.is_dir():
                    target = root / relative_path(member.filename)
                    with bundle.open(member) as stream:
                        expected = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest(target) != expected:
                        raise ValueError(f"Modified cached scene: {target}")
    else:
        entry = install(archive, manifest, cache)
    record = {**config, "usd_path": str(entry)}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=record_path.parent, delete=False) as stream:
        json.dump(record, stream, indent=2)
        temporary = Path(stream.name)
    temporary.replace(record_path)
    return entry


def load_record(path: Path) -> dict:
    """Read an offline scene record; fail before simulation if the entrypoint is missing."""
    record = json.loads(path.read_text())
    if record["schema_version"] != 1 or not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
        raise ValueError("Invalid scene record schema or revision")
    if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError("Invalid scene checksum")
    entry = Path(record["usd_path"])
    if not entry.is_absolute() or not entry.is_file():
        raise FileNotFoundError(f"Prepared scene is missing: {entry}; run foragebench.py fetch --config")
    return record


def catalog_scenes(repo_id: str, revision: str) -> tuple[str, list[dict]]:
    """Resolve one repository snapshot and read its scene manifests."""
    from huggingface_hub import HfApi, hf_hub_download

    commit = HfApi().dataset_info(repo_id, revision=revision).sha
    catalog = json.loads(
        Path(hf_hub_download(repo_id, "catalog.json", repo_type="dataset", revision=commit)).read_text()
    )
    if catalog["schema_version"] != 1:
        raise ValueError("Unsupported catalog schema")
    scenes = []
    identities = set()
    for name in catalog["scenes"]:
        path = str(relative_path(name))
        manifest = json.loads(Path(hf_hub_download(repo_id, path, repo_type="dataset", revision=commit)).read_text())
        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported scene schema")
        identity = (manifest["scene_id"], manifest["scene_version"])
        if not re.fullmatch(r"[a-z0-9_]+", identity[0]) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", identity[1]):
            raise ValueError(f"Invalid scene identity: {identity}")
        if identity in identities:
            raise ValueError(f"Duplicate catalog identity: {identity}")
        identities.add(identity)
        scenes.append({"manifest_path": path, "manifest": manifest})
    return commit, scenes


def select_scenes(scenes: list[dict], scene_ids: list[str] | None, version: str | None) -> list[dict]:
    """Select complete releases; reject missing or ambiguous scene versions."""
    if scene_ids is None:
        return scenes
    selected = []
    for scene_id in dict.fromkeys(scene_ids):
        matches = [
            s
            for s in scenes
            if s["manifest"]["scene_id"] == scene_id and (version is None or s["manifest"]["scene_version"] == version)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one release for {scene_id}, found {len(matches)}; use list and --scene_version")
        selected.extend(matches)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch")
    list_parser = commands.add_parser("list")
    list_parser.add_argument("--repo_id", default=REPO)
    list_parser.add_argument("--revision", default="main")
    fetch_parser.add_argument("--repo_id", default=REPO)
    selection = fetch_parser.add_mutually_exclusive_group()
    selection.add_argument("--scene_id", nargs="+")
    selection.add_argument("--all", action="store_true")
    fetch_parser.add_argument("--scene_version")
    fetch_parser.add_argument("--config", type=Path)
    fetch_parser.add_argument("--record", type=Path)
    fetch_parser.add_argument("--revision", help="Full dataset commit SHA")
    fetch_parser.add_argument("--manifest")
    fetch_parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("~/.cache/phygs/scenes").expanduser(),
        help="Default: $PHYGS_SCENES' default",
    )
    args = parser.parse_args()
    if args.command == "list":
        commit, scenes = catalog_scenes(args.repo_id, args.revision)
        print(f"Revision: {commit}")
        for scene in scenes:
            manifest = scene["manifest"]
            print(f"{manifest['scene_id']} {manifest['scene_version']} {manifest['archive_bytes']} bytes")
        return
    if args.scene_id or args.all:
        if args.config or args.manifest or args.record or (args.all and args.scene_version):
            parser.error(
                "Catalog selection cannot use --config/--manifest/--record; --scene_version requires --scene_id"
            )
        commit, scenes = catalog_scenes(args.repo_id, args.revision or "main")
        selected = select_scenes(scenes, args.scene_id, args.scene_version)
        for scene in selected:
            manifest = scene["manifest"]
            config = {
                key: manifest[key] for key in ("schema_version", "scene_id", "scene_version", "sha256", "physics")
            }
            config.update(repo_id=args.repo_id, revision=commit, manifest=scene["manifest_path"])
            record = args.cache_dir / "records" / commit / f"{manifest['scene_id']}-{manifest['scene_version']}.json"
            print(fetch_manifest(config, manifest, args.cache_dir, record))
            print(f"Scene record: {record.resolve()}")
            # The scene registry (phygs.episodes.scenes) finds the release by its id through this record.
            shutil.copyfile(record, args.cache_dir / f"{manifest['scene_id']}.json")
        print(f"Resolved revision: {commit}; fetched {len(selected)} scene releases")
        return
    if args.scene_version:
        parser.error("--scene_version requires --scene_id")
    if args.command == "fetch" and args.config:
        if args.revision or args.manifest or not args.record:
            parser.error("--config requires --record and cannot be combined with --revision/--manifest")
        print(fetch_config(args.config, args.cache_dir, args.record))
        return
    if args.command == "fetch" and (not args.revision or not args.manifest or args.record):
        parser.error("Provide --config with --record, or --revision with --manifest")
    from huggingface_hub import hf_hub_download

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Use a full immutable dataset commit SHA")
    manifest_path = hf_hub_download(
        args.repo_id, str(relative_path(args.manifest)), repo_type="dataset", revision=args.revision
    )
    manifest = json.loads(Path(manifest_path).read_text())
    archive = hf_hub_download(
        args.repo_id, str(relative_path(manifest["archive"])), repo_type="dataset", revision=args.revision
    )
    print(install(Path(archive), manifest, args.cache_dir))


if __name__ == "__main__":
    main()
