# config.py
# 此文件存储全局配置变量，方便统一管理和修改

import os

# ============= Hugging Face 相关配置 =============
# HF镜像站点地址，用于加速国内访问
HF_ENDPOINT = "https://hf-mirror.com"
# 禁用符号链接警告
HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
# 禁用hf_transfer下载方式，设为0表示禁用，1表示启用
HF_HUB_ENABLE_HF_TRANSFER = "0"
# 是否启用离线模式，设为0表示不启用，1表示启用离线模式
TRANSFORMERS_OFFLINE = "0"

# ============= CUDA相关配置 =============
# CUDA安装目录路径，用于PyTorch与CUDA的集成
CUDA_HOME = "D:\\CUDA"
CUDA_PATH = "D:\\CUDA"
# PyTorch安装命令
PYTORCH_INSTALL_CMD = "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121"

# ============= 模型配置 =============
# 默认使用的LLM模型名称
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ============= 路径配置 =============
def get_default_download_dir(model_name=DEFAULT_MODEL_NAME):
    """
    获取默认下载目录
    
    Args:
        model_name: 模型名称，格式可以是"组织/模型名"或"模型名"
        
    Returns:
        str: 模型下载保存的完整路径
    """
    # 从模型名称中提取组织和模型部分
    parts = model_name.split('/')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(parts) > 1:
        org, model = parts
        # 在项目主目录下创建models目录存放模型
        return os.path.join(base_dir, "models", org, model)
    else:
        # 在项目主目录下创建models目录存放模型
        return os.path.join(base_dir, "models", model_name)

# ============= 模型加载配置 =============
# 是否信任远程代码，设置为True允许模型执行随模型文件提供的Python代码
MODEL_TRUST_REMOTE_CODE = True
# 是否使用快速分词器，设置为False表示使用Python实现的分词器，可能更可靠但更慢
MODEL_USE_FAST_TOKENIZER = True
# 是否使用低内存加载模式，有助于在内存受限的环境中加载大模型
MODEL_LOW_CPU_MEM_USAGE = True
# 设备映射策略，"auto"表示自动选择最佳设备配置,"cuda:0"表示指定使用GPU设备
MODEL_DEVICE_MAP = "cuda:0"

# ============= 量化配置 =============
# 默认量化输出目录
QUANTIZE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "quantized")

# 量化类型选项
QUANTIZE_TYPES = {
    "dynamic_int8": "PyTorch动态INT8量化（Windows完全兼容）",
    "dynamic_fp16": "Float16半精度量化（Windows完全兼容）",
    "ggml_q4": "GGML Q4量化（需要安装ctransformers库）",
    "ggml_q8": "GGML Q8量化（需要安装ctransformers库）"
}

# 默认量化类型
DEFAULT_QUANTIZE_TYPE = "dynamic_int8"

# 性能测试设置
BENCHMARK_DEFAULT_PROMPTS = [
    "请介绍一下人工智能的发展历史和主要应用领域。",
    "什么是深度学习？它与传统机器学习有什么区别？",
    "解释一下大语言模型是如何工作的，以及它们面临的主要挑战。",
    "请为一个电商网站编写一个产品推荐算法的伪代码。",
    "如何评估一个自然语言处理模型的性能？请详细说明。"
]

# 性能测试输出目录
BENCHMARK_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results")

# 量化依赖包
QUANTIZE_DEPENDENCIES = {
    "basic": ["torch", "transformers", "tqdm", "psutil"],
    "ggml": ["ctransformers", "huggingface_hub[cli]"]
}

# 其他配置可以在此继续添加

# Example placeholder:
# API_KEY = os.getenv("API_KEY", "default_key")
# DATABASE_URL = "your_database_url_here" 