import omni.usd
import omni.physx
import numpy as np
from pxr import UsdGeom, UsdPhysics, Gf, Usd

class VectorizedPhysicsCulling:
    def __init__(self, robot_prim_path: str, env_prim_path: str, active_radius: float = 3.0):
        self.robot_path = robot_prim_path
        self.env_path = env_prim_path
        self.active_radius = active_radius
        self.stage = omni.usd.get_context().get_stage()
        
        self.robot_prim = self.stage.GetPrimAtPath(self.robot_path)
        self.robot_xform = UsdGeom.Xformable(self.robot_prim)
        
        # Parallel arrays for fast access
        self.env_apis = []       # Stores UsdPhysics.RigidBodyAPI
        self.env_xforms = []     # Stores UsdGeom.Xformable
        
        self._initialize_caches()
        
        self._physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_physics_step)
        
        self.frame_counter = 0
        self.check_interval = 10

    def _get_world_translation(self, xformable: UsdGeom.Xformable) -> np.ndarray:
        """Helper to get world translation as a NumPy array."""
        time = Usd.TimeCode.Default()
        transform = xformable.ComputeLocalToWorldTransform(time)
        pos = transform.ExtractTranslation()
        return np.array([pos[0], pos[1], pos[2]])

    def _initialize_caches(self):
        """Builds the initial caches and NumPy arrays."""
        positions_list = []
        kinematic_states_list = []
        
        for prim in self.stage.Traverse():
            if prim.GetPath().HasPrefix(self.env_path) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(prim)
                xform = UsdGeom.Xformable(prim)
                
                self.env_apis.append(api)
                self.env_xforms.append(xform)
                positions_list.append(self._get_world_translation(xform))
                
                # Assume true if the attribute isn't explicitly authored yet
                attr = api.GetKinematicEnabledAttr()
                kinematic_states_list.append(attr.Get() if attr.Get() is not None else False)

        # Main NumPy arrays
        self.positions = np.array(positions_list) # Shape: (N, 3)
        self.current_states = np.array(kinematic_states_list, dtype=bool) # Shape: (N,)
        self.num_objects = len(self.env_apis)

    def _on_physics_step(self, dt: float):
        self.frame_counter += 1
        if self.frame_counter % self.check_interval != 0 or not self.robot_xform:
            return

        # 1. Update positions ONLY for currently dynamic objects (they might be moving)
        # We skip the USD read for anything that is currently sleeping/kinematic
        dynamic_indices = np.where(~self.current_states)[0]
        for idx in dynamic_indices:
            self.positions[idx] = self._get_world_translation(self.env_xforms[idx])

        # 2. Get robot position
        robot_pos = self._get_world_translation(self.robot_xform)

        # 3. Vectorized distance calculation
        # np.linalg.norm computes Euclidean distance across the (N, 3) array instantly
        distances = np.linalg.norm(self.positions - robot_pos, axis=1)

        # 4. Determine which objects *should* be kinematic
        target_states = distances > self.active_radius

        # 5. Find objects whose state needs to change
        changed_indices = np.where(target_states != self.current_states)[0]

        # 6. Apply USD attribute changes ONLY to the objects that flipped states
        for idx in changed_indices:
            new_state = bool(target_states[idx])
            self.env_apis[idx].GetKinematicEnabledAttr().Set(new_state)
            
        # 7. Update the state cache
        self.current_states = target_states.copy()

    def cleanup(self):
        self._physics_sub = None