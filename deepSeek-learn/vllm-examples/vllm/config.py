"""
vLLM配置文件，定义全局配置参数

包含模型路径、API设置、推理参数等配置项
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class ModelConfig:
    """模型配置"""
    # 默认使用本地模型路径
    DEFAULT_MODEL_PATH: str = "models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    # 备选模型列表
    AVAILABLE_MODELS: Dict[str, str] = {
        "deepseek-r1-distill-qwen-7b": "models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-coder": "deepseek-ai/deepseek-coder-7b-instruct",
        "deepseek-llm": "deepseek-ai/deepseek-llm-7b-chat"
    }
    
    # 获取完整的模型路径
    @staticmethod
    def get_model_path(model_name: Optional[str] = None) -> str:
        """获取模型路径，支持模型名称、模型ID或完整路径"""
        if not model_name:
            return ModelConfig.DEFAULT_MODEL_PATH
            
        # 如果是已知模型简写名称，使用映射
        if model_name in ModelConfig.AVAILABLE_MODELS:
            return ModelConfig.AVAILABLE_MODELS[model_name]
            
        # 否则返回原始输入（可能是完整路径或HF模型ID）
        return model_name


@dataclass
class APIConfig:
    """API服务器配置"""
    DEFAULT_HOST: str = "0.0.0.0"
    DEFAULT_PORT: int = 8000
    MAX_BATCH_SIZE: int = 64
    MAX_WAITING_TOKENS: int = 10000
    TIMEOUT: int = 600


@dataclass
class InferenceConfig:
    """推理配置"""
    DEFAULT_MAX_TOKENS: int = 2048
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 50
    DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.85
    DEFAULT_TENSOR_PARALLEL_SIZE: int = 1
    

@dataclass
class VLLMConfig:
    """vLLM全局配置"""
    MODEL: ModelConfig = ModelConfig()
    API: APIConfig = APIConfig()
    INFERENCE: InferenceConfig = InferenceConfig()
    
    # 配置文件版本
    VERSION: str = "0.1.0"


# 导出全局配置实例
CONFIG = VLLMConfig() 