import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.kit import SimulationApp

# ---------- 启动 Isaac Sim ----------
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.utils.types import ArticulationAction
import omni.isaac.motion_generation as mg
from pxr import UsdPhysics
import omni.usd

np.random.seed(42)

# ================== 配置 ==================
USD_PATH = "/workspace/isaaclab/scripts/interactive-search/spot_model/spot_arm_w_cam.usd"
WHOLE_SPOT_URDF = "/workspace/isaaclab/scripts/interactive-search/spot_model/spot.urdf"
WHOLE_SPOT_YAML = "/workspace/isaaclab/scripts/interactive-search/spot_model/whole_spot_arm/whole_spot.yaml"

SPOT_ROOT = "/spot"
BODY_PATH = "/spot/body"

JOINT_ORDER = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]

# 关节限位（度 -> 弧度）
JOINT_LIMITS = {
    "arm_sh0": (np.deg2rad(-150), np.deg2rad(180)),
    "arm_sh1": (np.deg2rad(-180), np.deg2rad(30)),
    "arm_el0": (np.deg2rad(0),    np.deg2rad(180)),
    "arm_el1": (np.deg2rad(-160), np.deg2rad(160)),
    "arm_wr0": (np.deg2rad(-105), np.deg2rad(105)),
    "arm_wr1": (np.deg2rad(-165), np.deg2rad(165)),
}

LINKS_TO_CHECK = [
    ("arm_link_sh0", "/spot/arm_link_sh0"),
    ("arm_link_sh1", "/spot/arm_link_sh1"),
    ("arm_link_el0", "/spot/arm_link_el0"),
    ("arm_link_el1", "/spot/arm_link_el1"),
    ("arm_link_wr0", "/spot/arm_link_wr0"),
    ("arm_link_wr1", "/spot/arm_link_wr1"),
    ("arm_link_fngr", "/spot/arm_link_fngr"),
]

# ================== 初始化 World & 机器人 ==================
world = World(physics_dt=1/240.0, rendering_dt=1/60.0)
world.scene.add_default_ground_plane()

# 加载 Spot USD 到指定 prim path
add_reference_to_stage(USD_PATH, SPOT_ROOT)
world.reset()

# 封装 Articulation
art = Articulation(SPOT_ROOT)
art.initialize()

stage = world.stage

print("✅ Articulation 初始化完成")
print("   DOFs:", art.num_dof)
print("   DOF names:", art.dof_names)

# ========== 关节索引映射 ==========
usd_dof_names = art.dof_names
name_to_idx = {n: i for i, n in enumerate(usd_dof_names)}
REMAP = np.array([name_to_idx[n] for n in JOINT_ORDER], dtype=int)

print("🔧 关节索引映射:")
for i, jn in enumerate(JOINT_ORDER):
    print(f"  {jn} -> USD index {REMAP[i]}")

# ================== （可选）打印当前 DOF 属性 ==================
props = art.dof_properties  # 这里仅仅读取，不再 set_dof_properties
print("\n📌 当前部分 DOF 属性示例（type, lower, upper, stiffness, damping）:")
for jn in JOINT_ORDER:
    idx = name_to_idx[jn]
    print(
        f"  {jn:<10} | type={props['type'][idx]} "
        f"lower={props['lower'][idx]:.3f} upper={props['upper'][idx]:.3f} "
        f"kp={props['stiffness'][idx]:.1f} kd={props['damping'][idx]:.1f}"
    )

print("\n⚠️ Isaac Sim 5.0 中不再使用 set_dof_properties，"
      "这里直接使用 USD 中保存的关节驱动增益（如果需要改增益，建议在界面 / Gain Tuner 里改完再保存 USD）。")

# ================== 禁用 Spot 自身碰撞（视需要） ==================
print("\n👻 禁用 Spot 自身碰撞（仅用于 FK 验证）...")
collision_count = 0
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if path_str.startswith(SPOT_ROOT) and prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        collision_count += 1
print(f"  ✅ 已禁用 {collision_count} 个碰撞体")

# ================== 创建 RigidPrim 用于读取 link 位姿 ==================
print("\n📦 创建 RigidPrim 对象...")
body_prim = RigidPrim(BODY_PATH)
link_prims = {}

for lula_name, usd_path in LINKS_TO_CHECK:
    try:
        link_prims[lula_name] = RigidPrim(usd_path)
        print(f"  ✅ {lula_name} -> {usd_path}")
    except Exception as e:
        print(f"  ❌ {lula_name}: {e}")

# ================== Lula 求解器 ==================
print("\n🤖 加载 Lula Kinematics Solver...")
kin_solver = mg.LulaKinematicsSolver(
    robot_description_path=WHOLE_SPOT_YAML,
    urdf_path=WHOLE_SPOT_URDF,
)

print("   ✅ Lula Kinematics Solver 初始化完成")

# ================== 工具函数 ==================
def pose_to_mat(p, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot_matrix(quat_wxyz)
    T[:3, 3] = p
    return T

def get_T_diff(T_usd, T_lula):
    T_err = T_usd @ np.linalg.inv(T_lula)
    dp = np.linalg.norm(T_err[:3, 3])
    dR = R.from_matrix(T_err[:3, :3])
    dangle = np.rad2deg(dR.magnitude())
    return dp, dangle

# ========= 关键：使用 ArticulationAction + apply_action 做位置控制 =========
def set_joint_positions_with_control(q_arm, max_steps=200, tolerance=1e-3):
    """
    使用 Isaac Sim 5.0 推荐接口：
      - 构造完整关节向量 q_full（长度 = art.num_dof）
      - 用 ArticulationAction(joint_positions=...) + art.apply_action()
      - 世界步进，直到误差收敛
    """
    # 先取当前完整关节角，避免影响其他 DOF
    q_full = art.get_joint_positions()
    q_full[REMAP] = q_arm

    action = ArticulationAction(joint_positions=q_full)

    print("    ▶ 下发关节目标位置（通过 ArticulationAction + apply_action）...")

    converged = False
    final_error = None
    q_actual = None

    for step in range(max_steps):
        # 每一步都重新 apply_action，保证控制器持续收到目标
        art.apply_action(action)
        world.step(render=False)

        # 每 20 步检查一次收敛
        if step % 20 == 19:
            q_actual = art.get_joint_positions()[REMAP]
            error = np.linalg.norm(q_actual - q_arm)

            if error < tolerance:
                print(f"    ✅ 第 {step+1} 步收敛 (关节误差={error*1000:.3f} mrad)")
                converged = True
                final_error = error
                break
            else:
                print(f"    -> 第 {step+1} 步: 误差={error*1000:.1f} mrad")

    if not converged:
        # 最后再量一次
        q_actual = art.get_joint_positions()[REMAP]
        final_error = np.linalg.norm(q_actual - q_arm)
        print(f"    ⚠️ 未在 {max_steps} 步内收敛，最终误差={final_error*1000:.1f} mrad")
        print(f"    目标: {np.round(q_arm, 3)}")
        print(f"    实际: {np.round(q_actual, 3)}")

    return final_error, q_actual

# ================== FK 对比函数 ==================
def compare_all_links(q_arm, label="Test"):
    print(f"\n{'='*70}")
    print(f"🔍 {label}")
    print(f"   目标关节角 (arm joints): {np.round(q_arm, 3)}")

    # 驱动手臂关节 & 等待收敛
    conv_err, q_actual = set_joint_positions_with_control(
        q_arm, max_steps=300, tolerance=5e-3
    )

    print(f"   实际关节角: {np.round(q_actual, 3)}")
    print(f"   关节收敛误差: {conv_err:.6f} rad ({conv_err*1000:.1f} mrad)")

    # 如果关节没收敛，FK 对比就没意义
    if conv_err > 0.01:  # 10 mrad 阈值
        print(f"   ❌ 关节未收敛到目标，跳过 FK 比较")
        return False, 999.0, 999.0

    print(f"{'='*70}")
    print(f"{'Link':<25} | {'|Δp|(mm)':>10} | {'Δdeg(°)':>10} | {'Status':>8}")
    print("-" * 70)

    # Body 位姿
    p_body, q_body = body_prim.get_world_pose()
    T_world_body = pose_to_mat(p_body, q_body)

    all_ok = True
    max_dp = 0.0
    max_da = 0.0

    for lula_name, _ in LINKS_TO_CHECK:
        if lula_name not in link_prims:
            continue

        # ===== USD FK（RigidPrim） =====
        p_link, q_link = link_prims[lula_name].get_world_pose()
        T_world_link = pose_to_mat(p_link, q_link)
        T_usd = np.linalg.inv(T_world_body) @ T_world_link

        # ===== Lula FK（用“实际”关节角 q_actual）=====
        p_lula, R_lula = kin_solver.compute_forward_kinematics(lula_name, q_actual)
        T_lula = np.eye(4)
        T_lula[:3, :3] = R_lula
        T_lula[:3, 3] = p_lula

        # 误差
        dp, da = get_T_diff(T_usd, T_lula)
        max_dp = max(max_dp, dp)
        max_da = max(max_da, da)

        threshold_pos = 0.01   # 10 mm
        threshold_ang = 5.0    # 5 deg

        if dp > threshold_pos or da > threshold_ang:
            status = "❌ FAIL"
            all_ok = False
            print(f"\033[91m{lula_name:<25} | {dp*1000:10.2f} | {da:10.2f} | {status}\033[0m")
        else:
            status = "✅"
            print(f"{lula_name:<25} | {dp*1000:10.2f} | {da:10.2f} | {status}")

    print("-" * 70)
    print(f"最大误差: |Δp|={max_dp*1000:.2f} mm, Δdeg={max_da:.2f}°")

    if all_ok:
        print("✅ RigidPrim + 物理仿真 FK 验证通过")
    else:
        print("❌ 存在超标误差")

    return all_ok, max_dp, max_da

# ================== 运行测试 ==================
print("\n" + "="*70)
print("🚀 开始 RigidPrim + 物理仿真 FK 验证 (Isaac Sim 5.0)")
print("="*70)

# 测试 1: Zero Pose
q_zero = np.zeros(len(JOINT_ORDER))
ok1, dp1, da1 = compare_all_links(q_zero, "Zero Pose (q=0)")

# 测试 2-4: 在关节限位内随机采样
def sample_q_within_limits():
    """在关节限位内安全采样（预留 10% 边界）"""
    q = np.zeros(len(JOINT_ORDER))
    for i, jname in enumerate(JOINT_ORDER):
        lower, upper = JOINT_LIMITS[jname]
        margin = (upper - lower) * 0.1
        safe_lower = lower + margin
        safe_upper = upper - margin
        q[i] = np.random.uniform(safe_lower, safe_upper)
    return q

results = [(q_zero, ok1, dp1, da1)]

for i in range(3):
    q_rand = sample_q_within_limits()
    ok, dp, da = compare_all_links(q_rand, f"Random Pose {i+1}")
    results.append((q_rand, ok, dp, da))

# ================== 结果汇总 ==================
print("\n" + "="*70)
print("📊 测试结果汇总")
print("="*70)

all_passed = all(r[1] for r in results)
max_error_pos = max(r[2] for r in results)
max_error_ang = max(r[3] for r in results)

print(f"总测试数: {len(results)}")
print(f"通过数: {sum(r[1] for r in results)}")
print(f"失败数: {sum(not r[1] for r in results)}")
print(f"最大位置误差: {max_error_pos*1000:.2f} mm")
print(f"最大角度误差: {max_error_ang:.2f}°")

if all_passed:
    print("\n🎉 物理仿真模式 FK 验证通过！")
    print("✅ 可以把 Lula IK + 这个控制接口接到真实抓取/操作 pipeline：")
    print("   1) q_target = lula_ik.solve(ee_pose)")
    print("   2) 构造 ArticulationAction(joint_positions=q_full)")
    print("   3) 在控制循环中反复 art.apply_action(action) + world.step()")
else:
    print("\n⚠️  物理模式仍有误差偏大：")
    print("   - 检查 Spot USD 中的关节增益 (stiffness/damping)")
    print("   - 检查 Lula YAML / URDF 与 USD 关节名、父子关系是否完全一致")
    print("   - 放宽阈值或增加收敛步数再看")

simulation_app.close()
