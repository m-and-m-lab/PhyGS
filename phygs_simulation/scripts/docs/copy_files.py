from glob import glob
import os
import shutil


if __name__ == '__main__':
    # input_paths = glob('outputs/indoors/*/camera.png')

    # for input_path in input_paths:
    #     scene = input_path.split('/')[-2]
    #     output_path = f'assets/indoors/{scene}.png'

    input_paths = glob('outputs/aprtment-25-yue_small_solve/*/frames/Image/camera_0/Image_0_0_0048_0.png')

    for input_path in input_paths:
        episode_name = input_path.split('/')[2]
        output_path = f'docs/assets/small/{episode_name}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(output_path)
        shutil.copy2(input_path, output_path)