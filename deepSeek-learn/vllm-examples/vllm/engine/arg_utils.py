"""
定义引擎参数类，用于配置LLM引擎
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch


@dataclass
class EngineArgs:
    """LLM引擎参数配置类。
    
    提供配置vLLM引擎的各种参数。
    
    Attributes:
        model: 模型名称或路径
        tensor_parallel_size: 用于张量并行的GPU数量
        dtype: 模型权重的数据类型
        trust_remote_code: 是否信任远程代码
        gpu_memory_utilization: 目标GPU内存利用率
        max_num_batched_tokens: 最大批处理tokens数量
    """
    
    model: str
    tensor_parallel_size: int = 1
    dtype: Optional[Union[str, torch.dtype]] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    tokenizer: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code_tokenizer: Optional[bool] = None
    gpu_memory_utilization: float = 0.85
    swap_space: int = 0
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_paddings: int = 256
    disable_log_stats: bool = False
    max_model_len: Optional[int] = None
    enable_lora: bool = False
    max_loras: int = 4
    max_lora_rank: int = 16
    lora_extra_vocab_size: int = 256
    lora_dtype: Optional[Union[str, torch.dtype]] = None
    max_cache_position_ids: Optional[int] = None
    block_size: int = 16
    seed: int = 0
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    
    def __post_init__(self):
        """参数验证与转换"""
        # 验证参数取值范围
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size必须大于0")
        if not 0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization必须在(0, 1]范围内")
        
        # 处理数据类型
        if isinstance(self.dtype, str):
            if self.dtype == "half" or self.dtype == "float16":
                self.dtype = torch.float16
            elif self.dtype == "bfloat16":
                self.dtype = torch.bfloat16
            elif self.dtype == "float" or self.dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(f"未知的dtype: {self.dtype}")

@dataclass
class AsyncEngineArgs(EngineArgs):
    """异步LLM引擎参数配置类。
    
    继承自EngineArgs，添加异步操作所需的参数。
    
    Attributes:
        engine_use_ray: 是否使用Ray进行并行计算
        disable_log_requests: 是否禁用请求日志
        max_log_len: 最大日志长度
    """
    
    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: int = 100 