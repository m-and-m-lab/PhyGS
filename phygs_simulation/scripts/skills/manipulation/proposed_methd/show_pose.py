import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import os

# 1. 加载数据 (保持不变)
try:
    data = np.load('microwave_open.npz', allow_pickle=True)['data'].item()
except KeyError:
    # 兼容另一种可能的 .npz 格式
    data = np.load('microwave_open.npz', allow_pickle=True)

pts = data['input_pc'] if 'input_pc' in data else data['input']
heatmap = data['heatmap']
# 检查 grasps 是否存在，如果不存在则创建一个空列表
grasps = data.get('cgn_grasps', [])


# 2. 准备点云 (与之前相同)
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
colors = cm.get_cmap('RdYlGn')(heatmap)[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

print("数据加载完毕，准备弹出可视化窗口...")

# 3. 准备抓取姿态的可视化 (与之前相同)
def grasp_to_lineset(T, color=[0, 0.6, 0], radius=0.005):
    """
    把4x4抓取矩阵转成粗的Open3D几何（线条用圆柱体模拟）
    radius 控制粗细
    """
    g_opening = 0.08
    half = g_opening / 2
    p_base1, p_base2 = np.array([half, 0, 0]), np.array([-half, 0, 0])
    p_tip1, p_tip2 = np.array([half, 0.04, 0]), np.array([-half, 0.04, 0])
    p_back = np.array([0, -0.02, 0])

    pts_local = np.vstack([p_base1, p_base2, p_tip1, p_tip2, p_back])
    pts_world = (T[:3, :3] @ pts_local.T + T[:3, 3:4]).T
    pairs = [[0, 2], [1, 3], [0, 1], [0, 4], [1, 4]]

    cylinders = []
    for a, b in pairs:
        start, end = pts_world[a], pts_world[b]
        axis = end - start
        height = np.linalg.norm(axis)
        if height < 1e-6:
            continue

        axis = axis / height
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        mesh.paint_uniform_color(color)

        # 构造旋转矩阵
        z = np.array([0, 0, 1])
        v = np.cross(z, axis)
        c = np.dot(z, axis)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

        mesh.rotate(R, center=(0, 0, 0))
        mesh.translate((start + end) / 2)
        cylinders.append(mesh)

    combined = o3d.geometry.TriangleMesh()
    for c in cylinders:
        combined += c
    return combined


# 创建一个包含所有要显示物体的 Python 列表
geometries = [pcd]

# 将前5个抓取姿态也加入列表
top_k = min(1, len(grasps))
for i in range(top_k):
    geometries.append(grasp_to_lineset(grasps[i], radius=0.001))  # 抓取线条更粗


# 一行代码弹出窗口！
o3d.visualization.draw_geometries(
    geometries,
    window_name="抓取姿态可视化 (按 Q 关闭)",
    width=1280,
    height=720,
    point_show_normal=False
)

print("✅ 可视化窗口已关闭。")