from glob import glob
from natsort import natsorted
import os
import bpy
import numpy as np
from infinigen.core.init import configure_cycles_devices
import imageio.v2 as imageio


if __name__ == '__main__':
    for i_input, input_path in enumerate(natsorted(glob('outputs/single_cabinet_articulated/*.blend'))):
        print(input_path)
        # bpy.ops.wm.open_mainfile(filepath=input_path)

        # cameras = [ob for ob in bpy.context.scene.objects if ob.type == 'CAMERA']
        # # [bpy.data.objects['camera_0_0'], bpy.data.objects['camera_0_1']]
        # camera = cameras[0]

        # bpy.context.scene.render.engine = 'CYCLES'
        # bpy.context.scene.cycles.samples = 128
        # bpy.context.scene.cycles.use_denoising = True
        # configure_cycles_devices()
        # bpy.context.scene.camera = cameras[0]
        # bpy.context.scene.render.image_settings.file_format = 'PNG'

        with imageio.get_writer(f'outputs/single_cabinet_articulated_images/{i_input:06d}.gif', mode='I') as writer:
            for angle in np.arange(0, 180, 10):
                # right_door = bpy.data.objects['cabinet_right_door']
                # right_door.rotation_euler[2] = -angle / 180 * np.pi
                # if 'cabinet_left_door' in bpy.data.objects:
                #     left_door = bpy.data.objects['cabinet_left_door']
                #     left_door.rotation_euler[2] = angle / 180 * np.pi
                # bpy.context.scene.render.filepath = f'outputs/single_cabinet_articulated_images/{i_input:06d}_{angle:03d}.png'
                # bpy.ops.render.render(write_still=True)

                image_path = f'outputs/single_cabinet_articulated_images/{i_input:06d}_{angle:03d}.png'
                image = imageio.imread(image_path)
                image = image[..., :3] * (image[..., [3]] > 0)
                writer.append_data(image)

    