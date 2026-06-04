from glob import glob
from natsort import natsorted
import os
import bpy
from infinigen.core.init import configure_cycles_devices


if __name__ == '__main__':
    for input_path in natsorted(glob('outputs/apartment-5-seq-tiny-1view/*/scene.blend')):
        bpy.ops.wm.open_mainfile(filepath=input_path)

        cameras = [ob for ob in bpy.context.scene.objects if ob.type == 'CAMERA']
        # [bpy.data.objects['camera_0_0'], bpy.data.objects['camera_0_1']]
        
        for i_camera, camera in enumerate(cameras):
            output_path = f'{os.path.dirname(input_path)}/blender_imgs/{i_camera:02d}.png'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(input_path, output_path)
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.cycles.samples = 128
            bpy.context.scene.cycles.use_denoising = True
            configure_cycles_devices()
            bpy.context.scene.camera = camera
            bpy.context.scene.render.filepath = output_path
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.ops.render.render(write_still=True)