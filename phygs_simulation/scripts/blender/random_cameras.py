from glob import glob
from natsort import natsorted
import os
import bpy
from infinigen.core.init import configure_cycles_devices
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util import blender as butil
from infinigen.core import placement
import infinigen_examples.constraints.util as cu


if __name__ == '__main__':
    for input_path in natsorted(glob('outputs/apartment-25-seq-tiny/*/scene.blend')):
        output_path = f'{os.path.dirname(input_path)}/blend.png'
        print(input_path, output_path)

        solve_state_path = input_path.replace('scene.blend', 'solve_state.json')
        state = State.load(solve_state_path)
        input(state)

        camera_rigs = [butil.get_collection('camrig.0')]

        def pose_cameras():
            nonroom_objs = [
                o.obj for o in state.objs.values() if t.Semantics.Room not in o.tags
            ]
            scene_objs = solved_rooms + nonroom_objs

            scene_preprocessed = placement.camera.camera_selection_preprocessing(
                terrain=None, scene_objs=scene_objs
            )

            solved_floor_surface = butil.join_objects(
                [
                    tagging.extract_tagged_faces(o, {t.Subpart.SupportSurface})
                    for o in solved_rooms
                ]
            )

            placement.camera.configure_cameras(
                camera_rigs,
                scene_preprocessed=scene_preprocessed,
                init_surfaces=solved_floor_surface,
                nonroom_objs=nonroom_objs,
                terrain_coverage_range=None,  # do not filter cameras by terrain visibility, even if nature scenetype configs request this
            )
            butil.delete(solved_floor_surface)
            return scene_preprocessed

        # bpy.ops.wm.open_mainfile(filepath=input_path)

        # cameras = [ob for ob in bpy.context.scene.objects if ob.type == 'CAMERA']
        # camera = cameras[0]

        input()

        # bpy.context.scene.render.engine = 'CYCLES'
        # bpy.context.scene.cycles.samples = 128
        # bpy.context.scene.cycles.use_denoising = True
        # configure_cycles_devices()
        # bpy.context.scene.camera = cameras[0]
        # bpy.context.scene.render.filepath = output_path
        # bpy.context.scene.render.image_settings.file_format = 'PNG'
        # bpy.ops.render.render(write_still=True)