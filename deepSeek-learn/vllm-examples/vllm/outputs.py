"""
定义vLLM生成结果的输出类
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CompletionOutput:
    """单个完成项的输出类。
    
    Attributes:
        text: 生成的文本
        token_ids: 生成的token ID列表
        logprobs: token的log概率
        finished: 生成是否已完成
        stop_reason: 停止生成的原因
    """
    
    text: str
    token_ids: List[int]
    logprobs: Optional[List[Dict[int, float]]] = None
    finished: bool = False
    stop_reason: Optional[str] = None


@dataclass
class RequestOutput:
    """请求的完整输出类。
    
    Attributes:
        request_id: 请求的唯一ID
        prompt: 原始提示文本
        prompt_token_ids: 提示的token ID列表
        outputs: 生成的完成项列表
        finished: 请求是否已完成
    """
    
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool
    
    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished})"
        ) 