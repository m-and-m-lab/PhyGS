# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare and publish immutable scene releases to the scene dataset on Hugging Face.

A release is one scene id at one version: a ZIP made by ``package_scene.py`` plus a manifest
(``scenes/<id>/<version>/scene.json``: archive path, SHA-256, sizes, USD entry point, physics
timestep). The dataset's ``catalog.json`` lists every manifest; ``phygs.scene.fetch`` reads it.
Releases are never replaced: a changed scene is a new version.

    # 1. Package (scene_dir holds scene.usdc, textures/, README.md, ATTRIBUTION.md).
    ./isaaclab.sh -p $PHYGS/scene_generation/usd_tools/package_scene.py \\
      --scene_dir /scenes/bear_loop_house_storage_revision --output /scenes/bear_loop_house_storage_revision.zip
    # 2. Manifest and catalog entry in a local release folder (a checkout of the dataset's metadata).
    ./isaaclab.sh -p $PHYGS/scene_generation/usd_tools/publish_scene.py prepare \\
      --archive /scenes/bear_loop_house_storage_revision.zip --release_dir /scenes/releases \\
      --scene_id bear_loop_house --version 1.1.0
    # 3. One dataset commit with the catalog, the manifest, the card and the ZIP (``hf auth login`` first).
    ./isaaclab.sh -p $PHYGS/scene_generation/usd_tools/publish_scene.py upload \\
      --archive /scenes/bear_loop_house_storage_revision.zip --release_dir /scenes/releases \\
      --scene_id bear_loop_house --version 1.1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

REPO = "mandmlabresearch/ForageBench"
PHYSICS_DT = 1 / 240  # The timestep [s] the scenes are validated at; the framework may run faster.


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def manifest_path(release_dir: Path, scene_id: str, version: str) -> Path:
    return release_dir / "scenes" / scene_id / version / "scene.json"


def load_catalog(release_dir: Path) -> dict:
    path = release_dir / "catalog.json"
    if not path.is_file():
        return {"schema_version": 1, "scenes": []}
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("scenes"), list):
        raise ValueError(f"{path}: expected catalog schema 1 with a scenes list")
    return catalog


def prepare(archive: Path, release_dir: Path, scene_id: str, version: str, physics_dt: float) -> Path:
    """Write the release's manifest and add it to the catalog; refuse a release that exists."""
    if not re.fullmatch(r"[a-z0-9_]+", scene_id) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected a snake_case scene id and a numeric major.minor.patch version")
    with ZipFile(archive) as bundle:
        members = bundle.infolist()
        roots = {relative_path(member.filename).parts[0] for member in members}
    if len(roots) != 1:
        raise ValueError(f"The archive must hold one top-level folder; found {sorted(roots)}")
    (folder,) = roots
    entry = f"{folder}/scene.usdc"
    if entry not in {member.filename for member in members}:
        raise ValueError(f"The archive has no {entry}")
    path = manifest_path(release_dir, scene_id, version)
    if path.exists():
        raise FileExistsError(f"{path} exists; a changed scene is a new version, never a replaced one")
    manifest = {
        "schema_version": 1,
        "scene_id": scene_id,
        "scene_version": version,
        "archive": f"scenes/{scene_id}/{version}/{archive.name}",
        "sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "uncompressed_bytes": sum(member.file_size for member in members),
        "entrypoint": entry,
        "units": "meters",
        "up_axis": "Z",
        "physics": {"backend": "physx", "timestep_seconds": physics_dt},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    catalog = load_catalog(release_dir)
    entry_path = path.relative_to(release_dir).as_posix()
    if entry_path not in catalog["scenes"]:
        catalog["scenes"].append(entry_path)
    (release_dir / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(path)
    return path


def upload(archive: Path, release_dir: Path, scene_id: str, version: str, repo: str) -> None:
    """One commit on the private dataset: catalog, this release's manifest, the card and the ZIP."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    info = api.dataset_info(repo)
    if not info.private:
        raise ValueError(f"{repo} must be private before uploading (asset licences are under review)")
    path = manifest_path(release_dir, scene_id, version)
    manifest = json.loads(path.read_text())
    if digest(archive) != manifest["sha256"]:
        raise ValueError("Archive checksum does not match the manifest; run prepare again")
    existing = set(api.list_repo_files(repo, repo_type="dataset", revision=info.sha))
    if manifest["archive"] in existing or path.relative_to(release_dir).as_posix() in existing:
        raise FileExistsError(f"{scene_id} {version} is already published; releases are immutable")
    catalog = load_catalog(release_dir)
    if path.relative_to(release_dir).as_posix() not in catalog["scenes"]:
        raise ValueError("The catalog does not list this release; run prepare first")
    operations = [
        CommitOperationAdd(path_in_repo="catalog.json", path_or_fileobj=str(release_dir / "catalog.json")),
        CommitOperationAdd(path_in_repo=path.relative_to(release_dir).as_posix(), path_or_fileobj=str(path)),
        CommitOperationAdd(path_in_repo=manifest["archive"], path_or_fileobj=str(archive)),
    ]
    card = release_dir / "README.md"
    if card.is_file():
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card)))
    commit = api.create_commit(
        repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {scene_id} {version}",
        parent_commit=info.sha,
    )
    print(f"Published revision: {commit.oid}\n{commit.commit_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "upload"):
        sub = commands.add_parser(name)
        sub.add_argument("--archive", type=Path, required=True, help="The ZIP from package_scene.py")
        sub.add_argument("--release_dir", type=Path, required=True, help="Local folder with catalog.json and scenes/")
        sub.add_argument("--scene_id", required=True)
        sub.add_argument("--version", required=True)
        if name == "prepare":
            sub.add_argument("--physics_dt", type=float, default=PHYSICS_DT, help="Validated timestep [s]")
        else:
            sub.add_argument("--repo_id", default=REPO)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.archive.resolve(strict=True), args.release_dir, args.scene_id, args.version, args.physics_dt)
    else:
        upload(args.archive.resolve(strict=True), args.release_dir, args.scene_id, args.version, args.repo_id)


if __name__ == "__main__":
    main()
