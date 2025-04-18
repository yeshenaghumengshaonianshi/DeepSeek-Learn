"""
vLLM工具函数集
"""

import uuid
import time
from typing import Dict, List, Any


def random_uuid() -> str:
    """生成随机UUID字符串"""
    return str(uuid.uuid4())


def get_time_ms() -> int:
    """获取当前时间戳（毫秒）"""
    return int(time.time() * 1000)


def format_gpu_stats(gpu_stats: Dict[str, Any]) -> Dict[str, Any]:
    """格式化GPU统计信息"""
    formatted_stats = {}
    
    # 格式化内存使用情况
    if "mem_used" in gpu_stats and "mem_total" in gpu_stats:
        used_gb = gpu_stats["mem_used"] / (1024 ** 3)
        total_gb = gpu_stats["mem_total"] / (1024 ** 3)
        formatted_stats["mem_used_gb"] = round(used_gb, 2)
        formatted_stats["mem_total_gb"] = round(total_gb, 2)
        formatted_stats["mem_used_percent"] = round(used_gb / total_gb * 100, 1)
    
    # 格式化利用率
    if "utilization" in gpu_stats:
        formatted_stats["utilization_percent"] = round(gpu_stats["utilization"], 1)
    
    return formatted_stats


def print_model_size(num_params: int) -> None:
    """打印模型大小信息"""
    if num_params >= 1e12:
        size_str = f"{num_params / 1e12:.1f}T"
    elif num_params >= 1e9:
        size_str = f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        size_str = f"{num_params / 1e6:.1f}M"
    else:
        size_str = f"{num_params / 1e3:.1f}K"
    
    print(f"模型包含 {size_str} 参数")


def chunked_list(lst: List, chunk_size: int) -> List[List]:
    """将列表分割为指定大小的块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size] 