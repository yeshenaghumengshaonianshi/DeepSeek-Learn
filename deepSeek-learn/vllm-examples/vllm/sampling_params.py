"""
定义采样参数类，控制LLM生成的行为
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class SamplingParams:
    """控制LLM生成行为的采样参数类。
    
    Attributes:
        temperature: 控制生成的随机性。较高的值会产生更多样的输出。
        top_p: 使用nucleus采样，仅考虑累计概率为top_p的tokens。
        max_tokens: 最大生成的token数量。
        stop: 生成终止的字符串列表。
        presence_penalty: 用于减少重复内容的惩罚系数。
        frequency_penalty: 基于频率的惩罚系数。
        top_k: 仅从概率最高的top_k个token中选择。
        ignore_eos: 是否忽略EOS token。
        logprobs: 返回的log概率数量。
        use_beam_search: 是否使用beam search而非采样。
        best_of: beam search中保留的候选数量。
    """
    
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 16
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    top_k: int = -1
    ignore_eos: bool = False
    logprobs: Optional[int] = None
    use_beam_search: bool = False
    best_of: int = 1
    
    def __post_init__(self):
        # 验证参数
        if self.temperature < 0.0:
            raise ValueError("temperature必须非负")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p必须在[0, 1]范围内")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens必须为正")
        if self.top_k == 0:
            raise ValueError("top_k=0无效，设置为-1表示禁用，或设置为正值")
        
        # 标准化stop参数
        if isinstance(self.stop, str):
            self.stop = [self.stop]

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "top_k": self.top_k,
            "ignore_eos": self.ignore_eos,
            "logprobs": self.logprobs,
            "use_beam_search": self.use_beam_search,
            "best_of": self.best_of,
        } 