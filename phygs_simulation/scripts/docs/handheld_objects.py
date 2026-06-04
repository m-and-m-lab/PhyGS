from glob import glob
import imageio.v2 as imageio
import os
import numpy as np


if __name__ == '__main__':
    with open('docs/08_handheld.md', 'r') as f:
        lines = f.readlines()
    
    images = []
    for i, line in enumerate(lines):
        loc = line.find('assets/indoor_meshes')
        if loc == -1:
            continue

        image_path = 'docs/' + line[loc:-2]
        print(image_path)
        images.append(imageio.imread(image_path)[::2, ::2])
    
    images = np.concatenate([
        np.concatenate(images[:5], axis=1),
        np.concatenate(images[5:10], axis=1),
        np.concatenate(images[10:15], axis=1),
        np.concatenate(images[15:20], axis=1),
        np.concatenate(images[20:], axis=1),
    ], axis=0)

    os.makedirs('docs/handheld', exist_ok=True)
    imageio.imwrite('docs/handheld/handheld.png', images)