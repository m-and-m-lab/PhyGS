# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch the Ridgeback-Franka USD (NVIDIA's Isaac Sim 6.0 assets; Clearpath's BSD-3-Clause licence in
``assets/LICENSE``) into ``assets/``, and check every file against its pinned hash. Run once::

    python3 scripts/PhyGS/phygs/robots/ridgeback_franka/fetch.py

Standard library only; files already there with the right hash are kept.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

SOURCE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Clearpath/"
    "RidgebackFranka"
)
ASSETS = Path(__file__).resolve().parent / "assets"
FILES = {  # Path below SOURCE -> sha256 (fetched 2026-10-28).
    "LICENSE": "c5b036522ecf21b207daad625d10aff273d170b5cb46e3cae89129f67a6603c7",
    "Materials/LICENSE": "c5b036522ecf21b207daad625d10aff273d170b5cb46e3cae89129f67a6603c7",
    "Materials/Materials.usd": "163ad21137370dfa8f8fda6337fa5a84aab3b7dbcdd134c91792239272c33691",
    "Props/panda_hand.usd": "9bead1ccc72d702d333074ae30e5352f4328290e7e64dbf072696f052ba12a90",
    "Props/panda_leftfinger.usd": "53725ac396c13d8f90960780626c00ef05acdf1d1a6043f7bd7bd879750e6270",
    "Props/panda_link0.usd": "4b46e614cf37f745ef5f8212137ac6af06aaef34b3858ae02f2f67d2f2c988a7",
    "Props/panda_link1.usd": "9f3f8b7d980865d2023ead815fd2faed9727af533959077b5700e28f6242187b",
    "Props/panda_link2.usd": "f05fd2f8ce2533f06bb6b5bf916a0e8e8b151e5d1f58e270ce5617507590b27e",
    "Props/panda_link3.usd": "bf5b30913f51811cd7f772651d25ead0f48ed37d994a25c9fbccb38c8479dbf8",
    "Props/panda_link4.usd": "9dc792ca5a4fc5506cec8b7be83332d121727c0f32e3b133ad5a8478ab6a10e0",
    "Props/panda_link5.usd": "4b67be8b886c6705455b389ce6be7d710c75187397a8607381f1515c7b1e626c",
    "Props/panda_link6.usd": "ca05c47f20b3711dea3a36aa27c70d9047ce8fbdf5945eb2fbdb4069cd76308c",
    "Props/panda_link7.usd": "2f18324e59305650cf650d24d37e5bbe04e8f7e781da6e309105f6ec99840e0f",
    "Props/panda_rightfinger.usd": "c156a8aa36191c5debacb1048fd17a0d6d1939403d936db2235c8fd4ebcec926",
    "configuration/ridgeback_franka_robot_schema.usd": (
        "1a042c1135a095e501a9d435d3c0d65a053b95e885bc05afec003c7d6b27cc14"
    ),
    "ridgeback_franka.usd": "301e8b842327dc3a0b051b03cff9de4db16e29e329e8f18d4bbf9ed75fd5a1ae",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name, digest in FILES.items():
        target = ASSETS / name
        if target.is_file() and sha256(target) == digest:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{SOURCE}/{name}", timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"{name}: the download does not match its pinned hash; the source changed.")
        target.write_bytes(data)
        print(f"fetched {name}")
    print(f"assets ready in {ASSETS}")


if __name__ == "__main__":
    main()
