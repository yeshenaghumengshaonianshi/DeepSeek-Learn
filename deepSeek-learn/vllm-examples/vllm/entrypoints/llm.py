"""
定义LLM类，作为vLLM的主要入口点
"""

from typing import Dict, List, Optional, Union

import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


class LLM:
    """LLM类是vLLM的主要入口点。
    
    它封装了LLMEngine并提供了更简单的接口用于生成文本。
    
    Args:
        model: 模型名称或路径
        tensor_parallel_size: 用于张量并行的GPU数量
        trust_remote_code: 是否信任远程代码
        gpu_memory_utilization: 目标GPU内存利用率
        max_cache_position_ids: KV缓存中的最大位置ID数量
        enable_lora: 是否启用LoRA
        max_num_batched_tokens: 最大批处理tokens数量
        max_num_seqs: 最大序列数量
        revisions: 模型修订版本
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        gpu_memory_utilization: float = 0.85,
        max_cache_position_ids: Optional[int] = None,
        enable_lora: bool = False,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        **kwargs,
    ) -> None:
        """初始化LLM对象"""
        engine_args = EngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            revision=revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_cache_position_ids=max_cache_position_ids,
            enable_lora=enable_lora,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.model_config = self.llm_engine.model_config
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        prompt_tokens: Optional[List[List[int]]] = None,
    ) -> List[RequestOutput]:
        """使用LLM生成文本
        
        Args:
            prompts: 单个提示或提示列表
            sampling_params: 采样参数
            prompt_tokens: 预先标记化的提示tokens
            
        Returns:
            生成结果列表
        """
        # 默认采样参数
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # 标准化输入
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # 生成请求ID
        request_ids = [f"request_{i}" for i in range(len(prompts))]
        
        # 添加生成请求
        results = []
        for i, prompt in enumerate(prompts):
            request_id = request_ids[i]
            prompt_token = prompt_tokens[i] if prompt_tokens else None
            self.llm_engine.add_request(
                request_id, prompt, sampling_params, prompt_token_ids=prompt_token
            )
        
        # 生成并获取结果
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for request_output in step_outputs:
                if request_output.finished:
                    results.append(request_output)
        
        # 按原始顺序排序结果
        sorted_results = sorted(
            results, key=lambda x: request_ids.index(x.request_id)
        )
        return sorted_results 