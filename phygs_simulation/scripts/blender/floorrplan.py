from glob import glob
import json
from matplotlib import pyplot as plt
from natsort import natsorted
import os
import bpy
import numpy as np
from infinigen.core.init import configure_cycles_devices
import cv2
import imageio.v2 as imageio


if __name__ == '__main__':
    for i_scene, input_path in enumerate(natsorted(glob('outputs/apartment-5-seq-tiny-25views-by_room3/*/scene.blend'))):
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
        
        objs_points = []
        for obj_name, obj in placeholders.items():
            dims = np.array(obj['dimensions'])
            loc = np.array(obj['location'])
            euler = np.array(obj['rotation_euler'])
            print(obj_name, dims, loc, euler)
            R = np.array([
                [np.cos(euler[2]), -np.sin(euler[2]), 0],
                [np.sin(euler[2]), np.cos(euler[2]), 0],
                [0, 0, 1]
            ])
            
            half_dims = R @ dims / 2
            obj_points = loc + np.array([
                [-half_dims[0], -half_dims[1], half_dims[2]],
                [half_dims[0], -half_dims[1], half_dims[2]],
                [half_dims[0], half_dims[1], half_dims[2]],
                [-half_dims[0], half_dims[1], half_dims[2]]
            ]) 
            objs_points.append(obj_points)
        
        objs_points = sorted(objs_points, key=lambda x: x[0, 2])
        objs_points = np.array(objs_points)
        objs_points[..., 1] = -objs_points[..., 1]  # flip y axis
        min_x, max_x, min_y, max_y = np.min(objs_points[:, :, 0]), np.max(objs_points[:, :, 0]), np.min(objs_points[:, :, 1]), np.max(objs_points[:, :, 1])

        img = np.zeros([2000, 2000, 3], dtype=np.uint8)
        img.fill(255)
        # draw all rectangles with random color
        for obj_points in objs_points:
            # scale to fit in image
            obj_points = obj_points.copy()
            scale = max(max_x - min_x, max_y - min_y)
            obj_points[:, 0] = (obj_points[:, 0] - min_x) / scale * img.shape[0]
            obj_points[:, 1] = (obj_points[:, 1] - min_y) / scale * img.shape[1]
            # draw polygon
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            cv2.fillPoly(img, [obj_points[..., :2].astype(np.int32)], color=color.tolist())

        os.makedirs('docs/assets/tiny_floorplan', exist_ok=True)
        imageio.imwrite(f'docs/assets/tiny_floorplan/{i_scene}.png', img)
