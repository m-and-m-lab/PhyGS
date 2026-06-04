import argparse
import pprint
from pathlib import Path
import os

from infinigen.core.sim import sim_factory as sf

def export_articulated_asset(existing_obj, asset_type_name, cabinet_params=None, copy_from_existing=True):
    export_format = 'usd'
    export_dir = './sim_exports'
    doors_dir = "/".join([export_dir, export_format, asset_type_name])

    next_idx = 1001
    current_doors = []
    if os.path.exists(doors_dir):
        current_doors = [int(i) for i in os.listdir(doors_dir)]
    if len(current_doors) > 0:
        next_idx = max(current_doors) + 1

    export_path, semantic_mapping = sf.spawn_simready(
        name=asset_type_name,
        seed=next_idx,
        exporter=export_format,
        export_dir=Path(export_dir),
        visual_only=True,
        existing_asset=existing_obj,
        cabinet_parts=cabinet_params,
        copy_from_existing=copy_from_existing,
    )

    #print(f"Exported to {export_path.resolve()}")
    return next_idx
