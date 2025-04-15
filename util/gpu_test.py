import torch
import time

def check_gpu():
    # 检查CUDA是否可用
    print(f"PyTorch版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        
        # 创建两个大矩阵并在GPU上进行矩阵乘法
        print("\n执行GPU矩阵乘法测试...")
        
        # 创建两个大矩阵，但尺寸适中避免内存不足
        size = 2000  # 调小矩阵尺寸
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # 在GPU上计时矩阵乘法
        torch.cuda.synchronize()
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU矩阵乘法耗时: {gpu_time:.4f}秒")
        
        # 在CPU上进行相同的计算作为对比
        print("\n执行CPU矩阵乘法测试...")
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        print(f"CPU矩阵乘法耗时: {cpu_time:.4f}秒")
        print(f"GPU加速比: {cpu_time / gpu_time:.2f}倍")
        
        # 验证结果一致性
        error = torch.abs(c_cpu - c.cpu()).max().item()
        print(f"CPU和GPU结果最大误差: {error}")
        
        return True, gpu_time, cpu_time
    else:
        print("GPU不可用，无法运行测试")
        return False, 0, 0

if __name__ == "__main__":
    check_gpu() 