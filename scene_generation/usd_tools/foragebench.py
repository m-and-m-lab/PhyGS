# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare and publish immutable ForageBench scene releases."""

import argparse
import hashlib
import json
import re
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


def prepare(scene_dir: Path, release_dir: Path, scene_id: str, version: str) -> None:
    """Write release metadata while keeping the ZIP in its scene directory."""
    if not re.fullmatch(r"[a-z0-9_]+", scene_id) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected a snake_case scene ID and a numeric major.minor.patch version")
    archive = scene_dir / f"{scene_dir.name}.zip"
    with ZipFile(archive) as bundle:
        size = sum(item.file_size for item in bundle.infolist())
    manifest = {
        "schema_version": 1,
        "scene_id": scene_id,
        "scene_version": version,
        "archive": f"scenes/{scene_id}/{version}/{archive.name}",
        "sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "uncompressed_bytes": size,
        "entrypoint": f"{scene_dir.name}/scene.usdc",
        "units": "meters",
        "up_axis": "Z",
        "physics": {"backend": "physx", "timestep_seconds": 1 / 240},
    }
    manifest_path = release_dir / f"scenes/{scene_id}/{version}/scene.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (release_dir / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "scenes": [manifest_path.relative_to(release_dir).as_posix()]}, indent=2)
        + "\n"
    )
    print(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--scene_dir", type=Path, required=True)
    prepare_parser.add_argument("--release_dir", type=Path, required=True)
    prepare_parser.add_argument("--scene_id", required=True)
    prepare_parser.add_argument("--version", required=True)
    upload_parser = commands.add_parser("upload")
    upload_parser.add_argument("--release_dir", type=Path, required=True)
    upload_parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.scene_dir, args.release_dir, args.scene_id, args.version)
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    if args.command == "upload":
        api = HfApi()
        info = api.dataset_info(REPO)
        if not info.private:
            raise ValueError("ForageBench must be private before uploading")
        catalog = json.loads((args.release_dir / "catalog.json").read_text())
        if len(catalog["scenes"]) != 1:
            raise ValueError("This initial publisher accepts exactly one scene release")
        manifest_path = args.release_dir / relative_path(catalog["scenes"][0])
        manifest = json.loads(manifest_path.read_text())
        if digest(args.archive) != manifest["sha256"]:
            raise ValueError("Archive checksum mismatch")
        existing = api.list_repo_files(REPO, repo_type="dataset", revision=info.sha)
        if "catalog.json" in existing:
            raise ValueError("Dataset already initialized; prepare an explicit catalog update before another release")
        paths = ["README.md", "catalog.json", catalog["scenes"][0]]
        operations = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(args.release_dir / p)) for p in paths]
        operations.append(CommitOperationAdd(path_in_repo=manifest["archive"], path_or_fileobj=str(args.archive)))
        commit = api.create_commit(
            REPO,
            repo_type="dataset",
            operations=operations,
            commit_message="Add Bear Loop house 1.0.0",
            parent_commit=info.sha,
        )
        print(f"Published revision: {commit.oid}\n{commit.commit_url}")


if __name__ == "__main__":
    main()
