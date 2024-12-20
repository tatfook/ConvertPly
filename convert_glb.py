import os
import trimesh
import numpy as np

import utils

def load_glb_with_trimesh(file_path):
    """Load a GLB file using trimesh and return the mesh or scene."""
    loaded_object = trimesh.load(file_path)
    if isinstance(loaded_object, trimesh.Scene):
        # If it's a Scene, we need to process each geometry separately
        meshes = list(loaded_object.geometry.values())
    else:
        # If it's a single Trimesh, wrap it in a list for uniform processing
        meshes = [loaded_object]
    return meshes

def extract_material_colors(mesh):
    """Extract material colors from the mesh."""
    material_colors = []
    if hasattr(mesh.visual, 'material'):
        material = mesh.visual.material
        if hasattr(material, 'diffuse') and isinstance(material.diffuse, (list, tuple, np.ndarray)):
            diffuse_color = material.diffuse[:3]  # Take only RGB
            material_colors.append(diffuse_color)
        elif hasattr(material, 'baseColorFactor'):
            base_color_factor = material.baseColorFactor[:3]  # Take only RGB
            material_colors.append(base_color_factor)
    return material_colors

def has_valid_texture(mesh):
    """Check if the mesh has a valid texture."""
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        if hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                return True
            if hasattr(material, 'diffuseTexture') and material.diffuseTexture is not None:
                return True
    return False

def assign_material_colors_to_vertices(mesh, material_colors):
    """Assign material colors to vertices based on their associated materials."""
    if not hasattr(mesh.visual, 'material') or has_valid_texture(mesh):
        return
    
    vertex_colors = np.ones((len(mesh.vertices), 3), dtype=np.float32)
    
    if 'face_material' in mesh.metadata:
        for face_index, material_index in enumerate(mesh.metadata['face_material']):
            if material_index >= len(material_colors):
                continue
            material_color = material_colors[material_index]
            face = mesh.faces[face_index]
            vertex_colors[face] = material_color
    else:
        # If there's no face_material metadata, assume all faces use the first material
        if material_colors:
            material_color = material_colors[0]
            vertex_colors[:] = material_color
    
    mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

def save_mesh_as_glb(meshes, output_file_path):
    """Save the modified meshes as a new GLB file using trimesh."""
    if len(meshes) == 1:
        mesh = meshes[0]
    else:
        # Combine multiple meshes into a single scene
        scene = trimesh.scene.Scene()
        for i, mesh in enumerate(meshes):
            scene.add_geometry(mesh, node_name=f'mesh_{i}')
        mesh = scene
    mesh.export(output_file_path, file_type='glb')

def convert_model(input_file_path, output_file_path):
    # Load the GLB file
    meshes = load_glb_with_trimesh(input_file_path)

    for mesh in meshes:
        # Extract material colors
        material_colors = extract_material_colors(mesh)

        # Assign material colors to vertices only if there is no valid texture
        assign_material_colors_to_vertices(mesh, material_colors)

    # Save the modified mesh(es) as a new GLB file
    save_mesh_as_glb(meshes, output_file_path)

def convert_models(models_dir, output_dir = None):
    output_dir = output_dir or models_dir
    if (output_dir != models_dir):
        utils.delete_directory(output_dir)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)    
    
    models = utils.get_glb_model_files(models_dir)
    for model in models:
        base_name = os.path.splitext(os.path.basename(model))[0]
        output_file = os.path.join(output_dir, base_name + '.glb')
        convert_model(model, output_file)
        print(f"Converted {model} to {output_file}")
        
if __name__ == "__main__":
    convert_models("D:/workspace/npl/ConvertPly/models", "D:/workspace/npl/ConvertPly/models/outputs/")
    # input_file_path = "D:/workspace/npl/ConvertPly/models/deer.glb"
    # output_file_path = "D:/workspace/npl/ConvertPly/models/deer_colors.glb"
    # input_file_path = "D:/workspace/npl/ConvertPly/models/monitor_from_poly_by_google.glb"
    # output_file_path = "D:/workspace/npl/ConvertPly/models/monitor_from_poly_by_google_colors.glb"
    # convert_model(input_file_path, output_file_path)


# 将材质属性颜色赋值给顶点颜色
