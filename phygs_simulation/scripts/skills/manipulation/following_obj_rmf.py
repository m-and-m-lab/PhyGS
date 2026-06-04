# Fixed version of RMF control for Spot robot
import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import Articulation
from typing import Optional
import isaacsim.core.api.tasks as tasks
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.prims import Articulation
import omni.usd

from pxr import UsdGeom, Gf
from scipy.spatial.transform import Rotation as R
from isaacsim.core.prims import RigidPrim

# ---------------------------------------------------
def wxyz_to_xyzw(qwxyz):
    """Convert quaternion from wxyz to xyzw format"""
    return np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=float)

def pose_to_mat(p, quat_wxyz):
    """Convert pose to transformation matrix"""
    T = np.eye(4)
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    T[:3, :3] = quat_to_rot_matrix(quat_wxyz)
    T[:3, 3] = p
    return T

def get_link_pose(link_prim: RigidPrim):
    """Get world pose of a link"""
    p, q_wxyz = link_prim.get_world_poses()
    return p[0], q_wxyz[0] 

def check_joint_convergence(q_target, q_actual_full, remap, tolerance=5e-3):
    """Check if joints have converged to target positions"""
    q_actual_arm = q_actual_full[remap]
    error = np.linalg.norm(q_target - q_actual_arm)
    return error, (error < tolerance)

def print_ee_pose(ee_prim: RigidPrim, label="EE Pose"):
    """Print end effector pose"""
    p, q_wxyz = get_link_pose(ee_prim)
    print(f"{label}: p = {p}, quat(wxyz) = {q_wxyz}")
    
def calculate_look_at_quat(ee_pos, target_pos, forward_axis='+x'):
    direction = target_pos - ee_pos
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return np.array([0, 0, 0, 1]) # 保持原样
        
    v_forward = direction / dist 
    v_world_up = np.array([0, 0, 1])
    if np.abs(np.dot(v_forward, v_world_up)) > 0.99:
        v_world_up = np.array([1, 0, 0]) 

    v_right = np.cross(v_forward, v_world_up) 
    
    x_axis = v_forward
    y_axis = np.cross(v_world_up, x_axis) 
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    # Rotation Matrix = [X_col, Y_col, Z_col]
    rot_mat = np.column_stack((x_axis, y_axis, z_axis))
    r = R.from_matrix(rot_mat)
    return r.as_quat()    


class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        asset_path = (r"C:\Users\Guoji\Desktop\Files\isaac_spot\spot\spot.usd")
        add_reference_to_stage(usd_path=asset_path, prim_path="/spot")
        
        robot_prim_path = "/spot"
        end_effector_path = f"{robot_prim_path}/arm_link_fngr"
        
        # Define the gripper
        gripper = ParallelGripper(
            end_effector_prim_path=end_effector_path,
            joint_prim_names=["arm_f1x"], 
            joint_opened_positions=np.array([-1.2]),
            joint_closed_positions=np.array([0]),
            action_deltas=np.array([-0.05]),
            use_mimic_joints=True,
        )
        # Define the manipulator
        manipulator = SingleManipulator(
            prim_path=robot_prim_path,
            name="spot_robot",
            end_effector_prim_path=end_effector_path, 
            gripper=gripper,
            position=np.array([0.0, 0.0, 0.6]), 
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        return manipulator

class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:

        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(os.path.dirname(__file__), "../spot/whole_spot_arm/whole_spot.yaml"),
            rmpflow_config_path=os.path.join(os.path.dirname(__file__), "../spot/standalone_arm/spot_arm_rmpflow_common.yaml"),
            urdf_path=os.path.join(os.path.dirname(__file__), "../spot/whole_spot_arm/whole_spot.urdf"),
            end_effector_frame_name="spot_arm_link_fngr",
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


# Main execution
my_world = World(stage_units_in_meters=1.0)

# Initialize the Follow Target task
offset = np.array([0.0, 0.0, 0.0])
my_task = FollowTarget(
    name="spot_follow_target", 
    target_position=np.array([0.5, 0, 0.5]), 
    offset=offset
)
my_world.add_task(my_task)
my_world.reset()

# Get task parameters
task_params = my_world.get_task("spot_follow_target").get_params()
target_name = task_params["target_name"]["value"]
spot_name = task_params["robot_name"]["value"]
my_spot = my_world.scene.get_object(spot_name)

# Setup joint mapping
usd_dof_names = my_spot.dof_names
print(f"Available joints: {usd_dof_names}")

JOINT_ORDER = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]
name_to_idx = {n: i for i, n in enumerate(usd_dof_names)}
REMAP = np.array([name_to_idx[n] for n in JOINT_ORDER], dtype=int)
print("REMAP =", REMAP)

# Get articulation controller
articulation_controller = my_spot.get_articulation_controller()

# Initialize the RMPFlow controller
my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=my_spot)
my_controller.reset()

# Get end effector prim
ee_prim = RigidPrim("/spot/arm_link_wr1")
p_usd_batched, quat_usd_batched_wxyz = ee_prim.get_world_poses()
p_usd_world_current = p_usd_batched[0]
quat_usd_world_current_xyzw = wxyz_to_xyzw(quat_usd_batched_wxyz[0])
R_usd_world_current_rot = R.from_quat(quat_usd_world_current_xyzw)



# Run the simulation
while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()

        observations = my_world.get_observations()
        target_pos = observations[target_name]["position"]
        target_quat = observations[target_name]["orientation"]
        ee_pos_current, _ = ee_prim.get_world_poses()
        ee_pos_current = ee_pos_current[0] 
        look_at_quat_xyzw = calculate_look_at_quat(ee_pos_current, target_pos)
        target_quat_wxyz = np.array([
            look_at_quat_xyzw[3], # w
            look_at_quat_xyzw[0], # x
            look_at_quat_xyzw[1], # y
            look_at_quat_xyzw[2]  # z
        ])
        
        # Get actions from controller
        # Note: forward() expects wxyz format
        actions = my_controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=target_quat,  # wxyz format
        )

        if actions.joint_positions is not None:
            q_target = np.array(actions.joint_positions)
            
            # Check convergence
            q_actual = observations[spot_name]["joint_positions"]
            err, ok = check_joint_convergence(q_target, q_actual, REMAP)
            
            if my_world.current_time_step_index % 30 == 0:
                if ok:
                    print(f"✓ Joint convergence: error={err*1000:.3f} mrad")
                else:
                    print(f"✗ Joint not converged: error={err*1000:.3f} mrad")
                    
            articulation_controller.apply_action(actions)
        
        # Debug output
        if my_world.current_time_step_index % 150 == 0:
            # Print end effector to target distance
            ee_pos, _ = ee_prim.get_world_poses()
            ee_pos = ee_pos[0]
            target_pos = observations[target_name]["position"]
            dist = np.linalg.norm(ee_pos - target_pos)
            print(f"EE → Target distance: {dist:.4f} m")
            print_ee_pose(ee_prim, "End Effector (USD FK)")

simulation_app.close()