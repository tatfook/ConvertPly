# ConvertPly

## 环境搭建
1. 初始化venv环境 python -m venv .venv
2. 激活环境 .venv\Scripts\activate
3. 安装依赖包 pip install -r requirements.txt  
```bash 
# pip install numpy
# pip install open3d
# pip install plyfile
# pip install trimesh  直接奔溃 pip install trimesh[easy] 或 pip install trimesh[all]
# pip install pygltflib
```
4. 执行convert_glb 将glb的材质属性颜色放置顶点颜色中, open3d 无法解析获取材质属性颜色
5. 执行convert_ply 将glb文件转换为点云文件