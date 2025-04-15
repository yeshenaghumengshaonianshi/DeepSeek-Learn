import os
import sys
import subprocess

# 获取项目根目录并添加到模块搜索路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from config.config import CUDA_HOME, CUDA_PATH, PYTORCH_INSTALL_CMD

def run_command(cmd):
    """运行命令并打印输出"""
    print(f"执行: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    # 卸载现有PyTorch
    print("步骤1: 卸载现有PyTorch版本")
    run_command("pip uninstall torch torchvision torchaudio -y")
    
    # 设置CUDA环境变量
    os.environ["CUDA_HOME"] = CUDA_HOME
    os.environ["CUDA_PATH"] = CUDA_PATH
    print(f"CUDA_HOME设置为: {os.environ.get('CUDA_HOME')}")
    print(f"CUDA_PATH设置为: {os.environ.get('CUDA_PATH')}")
    
    # 使用磁盘上的CUDA 12.1安装PyTorch
    print("步骤2: 安装支持CUDA 12.1的PyTorch")
    run_command(PYTORCH_INSTALL_CMD)
    
    # 验证PyTorch安装
    print("步骤3: 验证PyTorch安装")
    verify_cmd = 'python -c "import torch; print(\'PyTorch版本:\', torch.__version__); print(\'CUDA可用:\', torch.cuda.is_available()); print(\'CUDA版本:\', torch.version.cuda); print(\'GPU名称:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'无可用GPU\')"'
    run_command(verify_cmd)

if __name__ == "__main__":
    main() 