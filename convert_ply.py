
import os
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
from scipy.spatial import KDTree
# import pyassimp

import utils

def load_valid_texture(mesh):
    # 遍历所有纹理，返回第一个有效纹理
    for texture in mesh.textures:
        if texture.is_empty():
            continue
        texture_image = np.asarray(texture)
        if texture_image.size > 0:
            return texture_image
    # 如果没有有效纹理，返回 None
    return None

# def assimp_model_to_o3d_mesh(model_file_path):
#     # 使用 pyassimp 加载模型
#     with pyassimp.load(model_file_path) as scene:
#         # 初始化 Open3D 的 TriangleMesh
#         o3d_mesh = o3d.geometry.TriangleMesh()

#         # 提取几何信息
#         for mesh in scene.meshes:
#             vertices = np.array(mesh.vertices, dtype=np.float32)
#             faces = np.array(mesh.faces, dtype=np.uint32)
            
#             # 将顶点添加到 Open3D 的 TriangleMesh
#             o3d_mesh.vertices.extend(o3d.utility.Vector3dVector(vertices))
            
#             # 将面添加到 Open3D 的 TriangleMesh
#             # 注意：Open3D 的面索引是从当前顶点列表的起始位置开始的
#             face_offset = len(o3d_mesh.vertices) - len(vertices)
#             o3d_mesh.triangles.extend(o3d.utility.Vector3iVector(faces + face_offset))
    
#     # 检查是否成功加载了顶点和面
#     if not o3d_mesh.has_vertices() or not o3d_mesh.has_triangles():
#         raise ValueError(f"{model_file_path} has no vertices or faces")
    
#     return o3d_mesh

def mesh_to_ply(mesh, ply_file_path, number_of_points=None, scale_factor=20):
    number_of_points = number_of_points or (scale_factor * scale_factor * scale_factor)  # 最大点默认为正方体体积

    # 将网格转换为点云
    pcd = mesh.sample_points_uniformly(number_of_points)  # 可以调整点的数量

    # 移除法线数据
    pcd.normals = o3d.utility.Vector3dVector()

    # 获取点云数据
    points = np.asarray(pcd.points)

    # TODO 应该根据三角形的材质ID来获取对象的纹理数据 triangle_material_ids
    texture_image = load_valid_texture(mesh)
    # 添加颜色信息（如果模型本身带有颜色）
    if mesh.has_vertex_colors():
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors)[:, :3]  # 取前三通道作为RGB

        # 创建KDTree以加速最近邻搜索
        kdtree = KDTree(vertices)
        
        # 获取采样点的颜色
        colors = []
        for point in pcd.points:
            dist, idx = kdtree.query(point)
            color = vertex_colors[idx]
            colors.append(color)

        # 将颜色转换为 numpy 数组
        colors = np.array(colors)
    elif texture_image is not None:
        # 获取网格顶点和三角面
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        triangle_uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)

        # 创建KDTree以加速最近邻搜索
        kdtree = KDTree(vertices)
        
        # 获取采样点的颜色
        colors = []
        for point in pcd.points:
            _, idx = kdtree.query(point)
            nearest_triangle = triangles[idx // 3]

            # 获取三角面顶点的UV坐标
            uv_coords = triangle_uvs[idx // 3]

            # 计算采样点的UV坐标（使用重心插值）
            v0, v1, v2 = vertices[nearest_triangle]
            uv0, uv1, uv2 = uv_coords

            # 使用最小二乘法计算重心坐标
            A = np.array([v0 - v2, v1 - v2]).T
            b = point - v2
            weight, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            weight = np.append(weight, 1 - weight.sum())
            uv = weight[0] * uv0 + weight[1] * uv1 + weight[2] * uv2
            
            # 确保UV坐标在[0,1]范围内
            uv = np.clip(weight[0] * uv0 + weight[1] * uv1 + weight[2] * uv2, 0, 1)

            # 将UV转换为纹理图像中的像素坐标
            u, v = uv
            x = int(u * (texture_image.shape[1] - 1))
            y = int((1 - v) * (texture_image.shape[0] - 1))  # 翻转V坐标
            color = texture_image[y, x, :3] / 255.0  # 归一化到[0, 1]
            colors.append(color)

        # 确保颜色数组与点云点数量匹配
        if len(colors) > len(pcd.points):
            colors = colors[:len(pcd.points)]
        elif len(colors) < len(pcd.points):
            extended_size = len(pcd.points) - len(colors)
            # 扩展颜色数组
            extended_colors = []
            while len(extended_colors) < extended_size:
                extended_colors.extend(colors)
            colors.extend(extended_colors)
            # 确保颜色数组长度与点云点数量一致
            colors = colors[:len(pcd.points)]
            
        # 将颜色转换为 numpy 数组
        colors = np.array(colors)
    else:
        # 如果没有纹理，生成全为1的颜色
        colors = np.ones((len(pcd.points), 3))

    # 将颜色值从 [0, 1] 转换为 [0, 255]
    colors = (colors * 255).astype(np.uint8)

    # 归一化点云数据到 [0, 1] 范围
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals
    normalized_points = (points - min_vals) / range_vals

    # 缩放点云数据并转换为整数
    scaled_points = (normalized_points * scale_factor).astype(np.int32)
    
    # 去除重复点
    unique_scaled_points = np.unique(scaled_points, axis=0)

    # 获取唯一点对应的索引
    _, unique_indices = np.unique(scaled_points, axis=0, return_index=True)
    unique_colors = colors[unique_indices]

    # 创建PLY元素
    vertex_element = np.array(
        [(x, z, y, r, g, b) for (x, y, z), (r, g, b) in zip(unique_scaled_points, unique_colors)],
        dtype=[
            ('x', 'i4'),
            ('y', 'i4'),
            ('z', 'i4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1')
        ]
    )

    # 创建PLY数据
    ply_data = PlyData([PlyElement.describe(vertex_element, 'vertex', comments=['vertices'])], text=True)
    
    # 写入PLY文件
    ply_data.write(ply_file_path)
    
    # 写入PLY文件
    # o3d.io.write_point_cloud(ply_file_path, pcd, write_ascii=True)

def model_to_o3d_mesh(model_file_path):
    mesh = o3d.io.read_triangle_mesh(model_file_path)
    mesh.compute_vertex_normals()

    return mesh

def convert_models(models_dir, output_dir = None):
    output_dir = output_dir or models_dir
    models = utils.get_model_files(models_dir)
    for model in models:
        base_name = os.path.splitext(os.path.basename(model))[0]
        output_file = os.path.join(output_dir, base_name + '.ply')

        try:
            if model.lower().endswith('.glb'):
                mesh = model_to_o3d_mesh(model)
            # elif model.lower().endswith('.fbx'):
                # mesh = assimp_model_to_o3d_mesh(model)
            else:
                mesh = model_to_o3d_mesh(model)
        except Exception as e:
            print(f"Error loading {model}: {e}")
            continue
    
        if len(mesh.triangles) == 0:
            print(f"Warning: The input mesh in {model} has no triangles. Skipping conversion.")
            continue
        
        mesh_to_ply(mesh, output_file)
        print(f"Converted {model} to {output_file}")


if __name__ == '__main__':
    # 调用一下函数 传入要转换的模型文件夹路径
    # convert_models('/mnt/d/workspace/program/ParacraftDev/worlds/DesignHouse/_user/wxatest/bmax/blocktemplates')
    # convert_models('D:/workspace/program/ParacraftDev/worlds/DesignHouse/_user/wxatest/bmax/blocktemplates')
    convert_models('D:/workspace/npl/ConvertPly/models')


# 使用示例
# glb_to_ply_open3d('/mnt/d/workspace/cmake-test/cache/models/deer.glb', '/mnt/d/workspace/cmake-test/cache/models/deer.ply')
# glb_to_ply_open3d('d:/workspace/cmake-test/cache/models/deer.glb', 'd:/workspace/cmake-test/cache/models/deer.ply')
# glb_to_ply_open3d('D:/workspace/program/ParacraftDev/worlds/DesignHouse/_user/wxatest/bmax/blocktemplates/deer.glb', 'D:/workspace/program/ParacraftDev/worlds/DesignHouse/_user/wxatest/bmax/blocktemplates/deer.ply')


# pip install numpy
# pip install open3d
# pip install plyfile
# pip install trimesh  直接奔溃 pip install trimesh[easy] 或 pip install trimesh[all]
# pip install pygltflib
# pip install pyassimp 不好使, 需要安装 assimp  ubuntu 可以使用 apt install libassimp-dev 安装


# 使用方法
# 1. 将需要转换的模型文件放入一个文件夹中，文件夹路径为 models_dir。
# 2. 将转换后的模型文件保存在另一个文件夹中，文件夹路径为 output_dir。
# 3. 调用 convert_models 函数，传入 models_dir 和 output_dir。
# 4. 函数会遍历 models_dir 中的所有文件，如果文件后缀为 .glb 或 .fbx，则将其转换为 .ply 文件，并将转换后的文件保存在 output_dir 中。

# 依赖库
# numpy: 用于处理数组和矩阵运算。
# open3d: 用于处理 3D 数据，如点云和三角网格。
# plyfile: 用于读取和写入 PLY 文件。
# trimesh: 用于处理 3D 模型，如加载和转换。
# pygltflib: 用于加载 GLTF 文件。
# pyassimp: 用于加载和转换 3D 模型文件

