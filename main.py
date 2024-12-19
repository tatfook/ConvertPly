
import os
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
# import pyassimp

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

def model_to_o3d_mesh(model_file_path):
    loaded_mesh = trimesh.load(model_file_path)
    
    if isinstance(loaded_mesh, trimesh.Scene):
        # 如果加载的是一个场景，提取所有几何对象并合并
        meshes = [loaded_mesh.geometry[geom] for geom in loaded_mesh.geometry]
        combined_mesh = trimesh.util.concatenate(meshes)
    else:
        combined_mesh = loaded_mesh

    if not combined_mesh.vertices.size or not combined_mesh.faces.size:
        raise ValueError(f"{model_file_path} has no vertices or faces")
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(combined_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(combined_mesh.faces)
    
    return o3d_mesh

def mesh_to_ply(mesh, ply_file_path, number_of_points=None, scale_factor=20):
    number_of_points = number_of_points or (scale_factor * scale_factor * scale_factor)  # 最大点默认为正方体体积

    # 将网格转换为点云
    pcd = mesh.sample_points_uniformly(number_of_points)  # 可以调整点的数量

    # 移除法线数据
    pcd.normals = o3d.utility.Vector3dVector()

    # 获取点云数据
    points = np.asarray(pcd.points)
    
    # 归一化点云数据到 [0, 1] 范围
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals
    normalized_points = (points - min_vals) / range_vals

    # 缩放点云数据并转换为整数
    scaled_points = (normalized_points * scale_factor).astype(np.int32)
    
    # 去除重复点
    unique_scaled_points = np.unique(scaled_points, axis=0)

    # 创建PLY元素
    vertex_element = np.array(
        [(x, z, y) for x, y, z in unique_scaled_points],
        dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'i4')]
    )

    # 创建PLY数据
    ply_data = PlyData([PlyElement.describe(vertex_element, 'vertex', comments=['vertices'])], text=True)
    
    # 写入PLY文件
    ply_data.write(ply_file_path)
    
    # 写入PLY文件
    # o3d.io.write_point_cloud(ply_file_path, pcd, write_ascii=True)


def get_model_files(directory):
    model_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.glb') or file.lower().endswith('.fbx'):
                model_files.append(os.path.join(root, file))

    return model_files


def convert_models(models_dir, output_dir = None):
    output_dir = output_dir or models_dir
    models = get_model_files(models_dir)
    for model in models:
        base_name = os.path.splitext(os.path.basename(model))[0]
        output_file = os.path.join(output_dir, base_name + '.ply')

        try:
            if model.lower().endswith('.glb'):
                mesh = model_to_o3d_mesh(model)
                # mesh = o3d.io.read_triangle_mesh(model)
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
# pip install trimesh  直接奔溃
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

