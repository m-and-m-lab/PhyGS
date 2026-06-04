from isaacsim import SimulationApp

# 你可以改成 headless=False 看窗口，其实这里不用仿真，只用 USD
simulation_app = SimulationApp({"headless": True})

import os
import math
import numpy as np
import xml.etree.ElementTree as ET

import omni.usd
from pxr import UsdGeom, Gf
from scipy.spatial.transform import Rotation as R


# ============ 配置区域：改成你自己的路径 ============
SPOT_USD_PATH  = "/workspace/isaaclab/scripts/interactive-search/spot_model/spot_arm_w_cam.usd"
URDF_PATH      = "/workspace/isaaclab/scripts/interactive-search/spot_model/spot.urdf"
SPOT_ROOT_PRIM = "/spot"   # 你的 spot 根 prim
# 只检查这几个关节（如果想要更多，可以自己加）
TARGET_JOINTS = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]
# ==================================================


def deg(v_rad):
    return np.array(v_rad) * 180.0 / np.pi


def print_vec(name, v):
    return f"{name}=[{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]"


def parse_urdf_origins(urdf_path):
    """
    解析 URDF，返回:
      joints[joint_name] = {
        "parent": parent_link,
        "child": child_link,
        "xyz": np.array([x,y,z]),
        "rpy": np.array([r,p,y])
      }
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    joints = {}
    for j in root.findall(f"{ns}joint"):
        name = j.attrib["name"]
        parent = j.find(f"{ns}parent").attrib["link"]
        child  = j.find(f"{ns}child").attrib["link"]
        origin_elem = j.find(f"{ns}origin")

        if origin_elem is not None:
            xyz_str = origin_elem.attrib.get("xyz", "0 0 0")
            rpy_str = origin_elem.attrib.get("rpy", "0 0 0")
        else:
            xyz_str = "0 0 0"
            rpy_str = "0 0 0"

        xyz = np.array([float(x) for x in xyz_str.split()], dtype=float)
        rpy = np.array([float(x) for x in rpy_str.split()], dtype=float)

        joints[name] = {
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
        }

    return joints


def link_name_to_prim_path(link_name):
    """
    把 URDF 里的 link 名字映射到 USD 里的 prim path。
    你的 URDF 用的是 spot_arm_link_sh0 / spot_body
    USD 是 /spot/arm_link_sh0 /spot/body

    这里做一个简单的规则：
      spot_body           -> /spot/body
      spot_arm_link_sh0   -> /spot/arm_link_sh0
      spot_arm_link_el0   -> /spot/arm_link_el0
      ...
    """
    if link_name == "base_link":
        return f"{SPOT_ROOT_PRIM}/body"

    if link_name.startswith("spot_"):
        # 去掉 "spot_" 前缀
        tail = link_name[len("spot_"):]
        return f"{SPOT_ROOT_PRIM}/{tail}"

    # 兜底：直接挂在 spot 下
    return f"{SPOT_ROOT_PRIM}/{link_name}"


def get_parent_child_world_T(stage, parent_path, child_path):
    """
    用 USD 的 Xformable 计算 parent->child 的变换：
      T_pc = T_world_parent^-1 * T_world_child
    """
    parent_prim = stage.GetPrimAtPath(parent_path)
    child_prim  = stage.GetPrimAtPath(child_path)

    if not parent_prim.IsValid():
        raise RuntimeError(f"Invalid parent prim path: {parent_path}")
    if not child_prim.IsValid():
        raise RuntimeError(f"Invalid child prim path: {child_path}")

    parent_xfable = UsdGeom.Xformable(parent_prim)
    child_xfable  = UsdGeom.Xformable(child_prim)

    T_world_parent = parent_xfable.ComputeLocalToWorldTransform(0.0)
    T_world_child  = child_xfable.ComputeLocalToWorldTransform(0.0)

    T_pc = T_world_parent.GetInverse() * T_world_child
    return T_pc


def decompose_gf_matrix(T_pc):
    """
    把 Gf.Matrix4d 分解成 (translation, rpy)
    rpy 顺序为 xyz（roll, pitch, yaw）
    """
    t = T_pc.ExtractTranslation()
    # ExtractRotation() return Gf.Rotation
    rot = T_pc.ExtractRotation()
    quat = rot.GetQuat()  # Gf.Quatd(real, imag)
    # convert to scipy Rotation
    q = np.array([quat.GetReal(), *quat.GetImaginary()], dtype=float)  # w, x, y, z
    # scipy needs (x, y, z, w)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R_mat = R.from_quat(q_xyzw)
    rpy = R_mat.as_euler("xyz", degrees=False)

    t_np = np.array([t[0], t[1], t[2]], dtype=float)
    return t_np, rpy


def main():
    print(">>> 启动 Isaac Sim 并加载 spot.usd ...")
    usd_ctx = omni.usd.get_context()
    usd_ctx.open_stage(SPOT_USD_PATH)
    stage = usd_ctx.get_stage()

    print(f">>> 解析 URDF: {URDF_PATH}")
    urdf_joints = parse_urdf_origins(URDF_PATH)

    print("\n=========== 对比 URDF vs USD 关节 origin ===========")

    for joint_name in TARGET_JOINTS:
        if joint_name not in urdf_joints:
            print(f"[WARN] Joint {joint_name} 不在 URDF 中，跳过")
            continue

        urdf_j = urdf_joints[joint_name]
        parent_link = urdf_j["parent"]
        child_link  = urdf_j["child"]
        xyz_urdf    = urdf_j["xyz"]
        rpy_urdf    = urdf_j["rpy"]

        parent_prim_path = link_name_to_prim_path(parent_link)
        child_prim_path  = link_name_to_prim_path(child_link)

        try:
            T_pc = get_parent_child_world_T(stage, parent_prim_path, child_prim_path)
        except RuntimeError as e:
            print(f"[ERROR] Joint {joint_name}: {e}")
            continue

        xyz_usd, rpy_usd = decompose_gf_matrix(T_pc)

        diff_xyz = xyz_usd - xyz_urdf
        diff_rpy = deg(rpy_usd - rpy_urdf)

        print(f"\n=== Joint: {joint_name} ===")
        print(f" parent link: {parent_link} -> prim: {parent_prim_path}")
        print(f" child  link: {child_link}  -> prim: {child_prim_path}")

        print(" URDF xyz:", xyz_urdf, " rpy(deg):", deg(rpy_urdf))
        print("  USD xyz:", xyz_usd, " rpy(deg):", deg(rpy_usd))

        print(" " + print_vec("Δxyz (m) ", diff_xyz))
        print(" " + print_vec("Δxyz (cm)", diff_xyz * 100.0))
        print(" " + print_vec("Δrpy (deg)", diff_rpy))

    print("\n=========== 对比结束 ===========")

    simulation_app.close()


if __name__ == "__main__":
    main()

