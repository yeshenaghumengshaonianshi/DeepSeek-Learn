"""
初始化项目目录结构
"""
import os
import sys

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def create_directory_structure():
    """创建项目所需的目录结构"""
    directories = [
        # 模型下载目录
        os.path.join(project_root, "models"),
        # 模型缓存目录
        os.path.join(project_root, "models", "offload_folder"),
        # 日志目录
        os.path.join(project_root, "logs"),
        # 数据目录
        os.path.join(project_root, "data"),
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
        else:
            print(f"目录已存在: {directory}")
    
    print("项目目录结构初始化完成!")

if __name__ == "__main__":
    create_directory_structure() 