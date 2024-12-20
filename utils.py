
import os
import shutil

def delete_directory(directory_path):
    """
    删除指定目录及其所有内容。
    
    参数:
    directory_path (str): 要删除的目录路径。
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        print(f"目录 {directory_path} 已成功删除")
    else:
        print(f"目录 {directory_path} 不存在")
        
def get_model_files(directory):
    model_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.glb') or file.lower().endswith('.fbx'):
                model_files.append(os.path.join(root, file))

    return model_files

def get_glb_model_files(directory):
    model_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.glb'):
                model_files.append(os.path.join(root, file))

    return model_files