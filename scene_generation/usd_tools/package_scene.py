#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package a portable USD scene and its actual dependencies into a verified ZIP."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pxr import UsdUtils


def dependency_files(scene_dir: Path) -> set[Path]:
    """Resolve the scene's complete dependency closure; reject missing or external files."""
    scene_dir = scene_dir.resolve(strict=True)
    entry = scene_dir / "scene.usdc"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(entry))
    if unresolved:
        raise ValueError(f"Unresolved USD dependencies: {unresolved}")
    files = {entry, *(Path(layer.realPath).resolve(strict=True) for layer in layers)}
    files.update(Path(asset).resolve(strict=True) for asset in assets)
    for path in files:
        if not path.is_relative_to(scene_dir):
            raise ValueError(f"Dependency is outside the scene directory: {path}")
        if not path.is_file():
            raise ValueError(f"Dependency is not a file: {path}")
    return files


def sha256(path: Path) -> str:
    """Compute a file checksum without loading the entire asset into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_scene(scene_dir: Path, output: Path) -> dict[str, int | str]:
    """Write and verify an archive containing runtime dependencies and distribution notes."""
    scene_dir = scene_dir.resolve(strict=True)
    output = output.resolve()
    if output.suffix != ".zip":
        raise ValueError("Output must be a .zip file.")
    files = dependency_files(scene_dir)
    for name in ("README.md", "ATTRIBUTION.md"):
        path = scene_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Distribution document must be a regular local file: {path}")
        files.add(path)
    files = sorted(files)
    checksums = {path.relative_to(scene_dir).as_posix(): sha256(path) for path in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene-package-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / output.name
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, f"{scene_dir.name}/{path.relative_to(scene_dir).as_posix()}")
            archive.writestr(
                f"{scene_dir.name}/SHA256SUMS",
                "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
            )
        extracted = temporary / "extracted"
        with ZipFile(archive_path) as archive:
            archive.extractall(extracted)
        extracted_scene = extracted / scene_dir.name
        # Re-resolving after extraction catches absolute paths back to the original directory.
        dependency_files(extracted_scene)
        for name, digest in checksums.items():
            if sha256(extracted_scene / name) != digest:
                raise ValueError(f"Archive checksum mismatch: {name}")
        digest = sha256(archive_path)
        archive_path.replace(output)
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n")
    return {
        "archive": str(output),
        "files": len(files) + 1,
        "uncompressed_bytes": sum(path.stat().st_size for path in files),
        "archive_bytes": output.stat().st_size,
        "sha256": digest,
    }


def main() -> None:
    """Package one scene without including generation tools, logs or unused assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key, value in package_scene(args.scene_dir, args.output).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
