from glob import glob
from natsort import natsorted
import os
import bpy
from infinigen.core.init import configure_cycles_devices


if __name__ == '__main__':
    for input_path in natsorted(glob('outputs/indoors/*/export_scene.blend/export_scene.usdc')):
        output_path = '/'.join(input_path.split('/')[:3]) + '/usd.png'
        print(input_path, output_path)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.usd_import(filepath=input_path)

        cameras = [ob for ob in bpy.context.scene.objects if ob.type == 'CAMERA']
        # [bpy.data.objects['camera_0_0'], bpy.data.objects['camera_0_1']]
        camera = cameras[0]

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.use_denoising = True
        configure_cycles_devices()
        bpy.context.scene.camera = cameras[0]
        bpy.context.scene.render.filepath = output_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)