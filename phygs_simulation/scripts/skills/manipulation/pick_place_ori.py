from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
import os
import numpy as np
from typing import Optional
import isaacsim.core.api.tasks as tasks
from isaacsim.core.prims import Articulation
from isaacsim.core.prims import SingleArticulation
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
import isaacsim.robot.manipulators.controllers as manipulators_controllers
import omni.usd
from isaacsim.core.utils.types import ArticulationAction
from isaaclab.assets import Articulation


from pxr import UsdGeom, Gf
from scipy.spatial.transform import Rotation as R
from isaacsim.core.prims import RigidPrim

# ---------------------------------------------------
def wxyz_to_xyzw(qwxyz):
    return np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=float)

def pose_to_mat(p, quat_wxyz):
    T = np.eye(4)
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    T[:3, :3] = quat_to_rot_matrix(quat_wxyz)
    T[:3, 3] = p
    return T

def get_link_pose(link_prim: RigidPrim):
    p, q_wxyz = link_prim.get_world_poses()
    return p[0], q_wxyz[0] 


def check_joint_convergence(q_target, q_actual_full, remap, tolerance=5e-3):
    q_actual_arm = q_actual_full[remap]  
    error = np.linalg.norm(q_target - q_actual_arm)
    return error, (error < tolerance)


def print_ee_pose(ee_prim: RigidPrim, label="EE Pose"):
    p, q_wxyz = get_link_pose(ee_prim)
    print(f"{label}: p = {p}, quat(wxyz) = {q_wxyz}")


class PickPlace(tasks.PickPlace):
    def __init__(
        self,
        name: str = "ur10e_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = np.array([0.0515, 0.0515, 0.0515]),
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        asset_path = "/workspace/isaaclab/scripts/interactive-search/spot_model/spot_arm_w_cam.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/spot")
        stage = omni.usd.get_context().get_stage()
        spot_prim = stage.GetPrimAtPath("/spot")

        xf = UsdGeom.XformCommonAPI(spot_prim)
        xf.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.5))
        xf.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        
        robot_prim_path = "/spot" 
        end_effector_path = f"{robot_prim_path}/arm_link_fngr"

        # define the gripper
        gripper = ParallelGripper(
            end_effector_prim_path=end_effector_path,
            joint_prim_names=["arm_f1x"], 
            joint_opened_positions=np.array([-1.2]),
            joint_closed_positions=np.array([-0.4]),
            action_deltas=np.array([-0.05]),
            use_mimic_joints=True,
        )
        # define the manipulator
        manipulator = SingleManipulator(
            prim_path=robot_prim_path,
            name="spot_robot",
            end_effector_prim_path=end_effector_path, 
            gripper=gripper,
            position=np.array([0.0, 0.0, 0.5]), 
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        return manipulator


class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:

        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            # robot_description_path=os.path.join(os.path.dirname(__file__), "../spot/whole_spot_arm/whole_spot.yaml"),
            # rmpflow_config_path=os.path.join(os.path.dirname(__file__), "../spot/standalone_arm/spot_arm_rmpflow_common.yaml"),
            # urdf_path=os.path.join(os.path.dirname(__file__), "../spot/whole_spot_arm/whole_spot.urdf"),
            robot_description_path="/workspace/isaaclab/scripts/interactive-search/spot_model/whole_spot_arm/whole_spot.yaml",
            rmpflow_config_path="/workspace/isaaclab/scripts/interactive-search/spot_model/whole_spot_arm/spot_arm_rmpflow_common.yaml",
            urdf_path="/workspace/isaaclab/scripts/interactive-search/spot_model/spot.urdf",
            end_effector_frame_name="arm_link_fngr",
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        
        
    def set_target(self, target_world_pose: np.ndarray, target_world_orientation: np.ndarray) -> None:

        if self.R_fix_rot is None or self.t_fix is None:
            p_rmp_g_world = target_world_pose
            quat_rmp_g_world = target_world_orientation
        else:
            p_usd_g_world = target_world_pose
            R_usd_g_world_rot = R.from_quat(target_world_orientation)
            
            R_rmp_g_world_rot = R_usd_g_world_rot * self.R_fix_rot.inv()
            p_rmp_g_world = p_usd_g_world - R_rmp_g_world_rot.apply(self.t_fix)
            quat_rmp_g_world = R_rmp_g_world_rot.as_quat()

        super().set_target(p_rmp_g_world, quat_rmp_g_world)


class PickPlaceController(manipulators_controllers.PickPlaceController):
    def __init__(
        self, name: str, gripper: ParallelGripper, robot_articulation: SingleArticulation, events_dt=None
    ) -> None:
        if events_dt is None:
            events_dt = [0.005, 0.005, 0.08, 0.006, 0.0005, 0.0005, 0.005, 0.008, 0.0008, 0.008]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt
        )
        return
    

my_world = World(stage_units_in_meters=1.0)

cube_initial_position = np.array([0.7, 0.2, 0.03/2.0])
cube_initial_orientation = np.array([0.7071, 0.0, 0.0, 0.7071])
target_position = np.array([0.7, -0.2, 0.03/2.0])

my_task = PickPlace(
    name="spot_arm_pick_place",
    cube_initial_position=cube_initial_position,
    cube_initial_orientation=cube_initial_orientation,
    target_position=target_position,
    cube_size=np.array([0.1, 0.03, 0.1])
)

my_world.add_task(my_task)
my_world.reset()

base_rigid = RigidPrim("/spot/body")

for _ in range(60):
    my_world.step(render=True)


task_params = my_world.get_task("spot_arm_pick_place").get_params()
spot_name = task_params["robot_name"]["value"]
my_spot = my_world.scene.get_object(spot_name)

usd_dof_names = my_spot.dof_names
JOINT_ORDER = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]

name_to_idx = {n: i for i, n in enumerate(usd_dof_names)}
REMAP = np.array([name_to_idx[n] for n in JOINT_ORDER], dtype=int)
print("REMAP =", REMAP)


my_controller = PickPlaceController(name="controller", robot_articulation=my_spot, gripper=my_spot.gripper)
cspace_controller = my_controller._cspace_controller
rmp = cspace_controller.rmpflow

ee_prim = RigidPrim("/spot/arm_link_fngr")
p_usd_batched, quat_usd_batched_wxyz = ee_prim.get_world_poses()
p_usd_world_current = p_usd_batched[0]

quat_usd_world_current_xyzw = wxyz_to_xyzw(quat_usd_batched_wxyz[0])
R_usd_world_current_rot = R.from_quat(quat_usd_world_current_xyzw)


usd_ee_quat_des = np.array([0.0, -0.707, 0.0, 0.707])  

articulation_controller = my_spot.get_articulation_controller()
reset_needed = True
    

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_playing():
        
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False

        if my_world.current_time_step_index == 0:
            my_controller.reset()

        observations = my_world.get_observations()
        
        pick_pos_usd = observations[task_params["cube_name"]["value"]]["position"]
        place_pos_usd = observations[task_params["cube_name"]["value"]]["target_position"]
        
        actions = my_controller.forward(
            picking_position=pick_pos_usd,
            placing_position=place_pos_usd,
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_orientation=usd_ee_quat_des, 
            end_effector_offset=np.array([0, 0.0, -0.4]),
        )
        if actions.joint_positions is not None:
            q_target = np.array(actions.joint_positions)
        else:
            q_target = None

        articulation_controller.apply_action(actions)

        if my_world.current_time_step_index % 150 == 0:
            print_ee_pose(ee_prim, "End Effector (USD FK)")
            ee_pos, _ = ee_prim.get_world_poses()
            ee_pos = ee_pos[0]
            pick_pos_usd = observations[task_params["cube_name"]["value"]]["position"]
            dist = np.linalg.norm(ee_pos - pick_pos_usd)
            print(f"EE → Cube distance: {dist:.4f} m")
            """
            q_actual = observations[spot_name]["joint_positions"]
            if q_target is not None:
                err, ok = check_joint_convergence(q_target, q_actual, REMAP)
                if ok:
                    print(f"✓ Joint convergence: error={err*1000:.3f} mrad")
                else:
                    print(f"✗ Joint not converged: error={err*1000:.3f} mrad")
            """



        if my_controller.is_done():
            print("✅ Task Done")
            reset_needed = True

    if my_world.is_stopped():
        reset_needed = True

simulation_app.close()