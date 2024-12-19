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
    
    # 提取纹理信息
    texture_images = {}
    texture_uv = {}

    if hasattr(combined_mesh.visual, 'material') and combined_mesh.visual.material is not None:
        material = combined_mesh.visual.material
        # 尝试获取 baseColorTexture
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            diffuse_texture = material.baseColorTexture
            if hasattr(diffuse_texture, 'image') and diffuse_texture.image is not None:
                texture_image = np.array(diffuse_texture.image)  # 将 PIL.Image 转换为 numpy 数组
                texture_images['diffuse'] = texture_image
                texture_uv['diffuse'] = combined_mesh.visual.uv
        # 尝试获取 diffuseTexture
        if hasattr(material, 'diffuseTexture') and material.diffuseTexture is not None:
            diffuse_texture = material.diffuseTexture
            if hasattr(diffuse_texture, 'image') and diffuse_texture.image is not None:
                texture_image = np.array(diffuse_texture.image)  # 将 PIL.Image 转换为 numpy 数组
                texture_images['diffuse'] = texture_image
                texture_uv['diffuse'] = combined_mesh.visual.uv

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(combined_mesh.vertices.astype(np.float64))
    
    # 确保 faces 是 (N, 3) 形状的整数数组
    if len(combined_mesh.faces) > 0:
        try:
            o3d_mesh.triangles = o3d.utility.Vector3iVector(combined_mesh.faces)
        except Exception as e:
            print(f"Error setting triangles: {e}")
            print(f"Faces data type: {combined_mesh.faces.dtype}")
            print(f"Faces shape: {combined_mesh.faces.shape}")
            raise
    else:
        print("No faces found in the mesh.")

    # 映射纹理到顶点上
    if 'diffuse' in texture_images and 'diffuse' in texture_uv:
        uv_coords = texture_uv['diffuse']
        texture_image = texture_images['diffuse']

        # 将 UV 坐标归一化到 [0, 1]
        uv_coords = uv_coords % 1.0

        # 计算纹理图像的宽度和高度
        height, width, _ = texture_image.shape

        # 将 UV 坐标映射到纹理图像上的像素坐标
        pixel_coords = (uv_coords * np.array([width, height])).astype(int)

        # 提取对应像素的颜色
        colors = texture_image[pixel_coords[:, 1], pixel_coords[:, 0]]

        # 将颜色值归一化到 [0, 1]
        colors = colors.astype(np.float64) / 255.0

        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return o3d_mesh