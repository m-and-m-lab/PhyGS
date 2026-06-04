from glob import glob
from natsort import natsorted
import os
import bpy
import numpy as np
from infinigen.core.init import configure_cycles_devices
import imageio.v2 as imageio
from natsort import natsorted


if __name__ == '__main__':
    for i_scene, scene_path in enumerate(natsorted(glob('outputs/apartment-5-seq-tiny-25views-by_room3/*'))):
        assert os.path.basename(scene_path) == str(i_scene)
        os.makedirs('docs/assets/tiny_25views/', exist_ok=True)
        with imageio.get_writer(f'docs/assets/tiny_25views/{i_scene}.gif', mode='I', fps=1) as writer:
            for image_path in natsorted(glob(f'{scene_path}/blender_imgs/*.png')):
                image = imageio.imread(image_path)
                image = image[..., :3] * (image[..., [3]] > 0)
                writer.append_data(image)

    