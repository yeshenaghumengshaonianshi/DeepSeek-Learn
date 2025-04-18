"""
vLLM: 高效的大型语言模型推理引擎
"""

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.llm import LLM
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

__version__ = "0.2.0"

__all__ = [
    "LLM",
    "LLMEngine",
    "AsyncLLMEngine",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "EngineArgs",
    "AsyncEngineArgs",
] 