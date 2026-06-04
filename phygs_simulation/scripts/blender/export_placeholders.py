from glob import glob
import json
from natsort import natsorted
import os
import bpy
from infinigen.core.init import configure_cycles_devices


if __name__ == '__main__':
    for input_path in natsorted(glob('outputs/apartment-5-seq-tiny-25views-by_room3/*/scene.blend')):
        bpy.ops.wm.open_mainfile(filepath=input_path)

        cameras = [ob for ob in bpy.context.scene.objects if ob.type == 'CAMERA']
        
        # collections
        collection_names = [col.name for col in bpy.data.collections]
        placeholders = {}
        for obj in bpy.data.collections['placeholders'].objects:
            placeholders[obj.name] = {
                'type': obj.type,
                'dimensions': [f for f in obj.dimensions],
                'location': [f for f in obj.location],
                'rotation_euler': [f for f in obj.rotation_euler],
            }
        
        output_path = input_path.replace('scene.blend', 'placeholders.json')
        with open(output_path, 'w') as f:
            json.dump(placeholders, f, indent=2)
        print(input_path, output_path)
        