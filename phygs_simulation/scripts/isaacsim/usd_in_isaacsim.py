from glob import glob
from natsort import natsorted
import os
import omni
import isaacsim
from omni.isaac.kit import SimulationApp

if __name__ == '__main__':
    simulation_app = SimulationApp(launch_config={"renderer": "RayTracedLighting", "headless": False})
    for input_path in natsorted(glob('outputs/indoors/*/export_scene.blend/export_scene.usdc')):
        output_path = '/'.join(input_path.split('/')[:3]) + '/usd.png'
        print(input_path, output_path)

        input_path = os.path.abspath(input_path)

        omni.usd.get_context().open_stage(input_path)
