"""
高效内存管理的DeepSeek模型量化脚本

本脚本专为Windows环境设计，使用分层量化技术减少内存占用。
支持多种量化方法，并自动处理兼容性问题。

使用方法: python deepSeek-learn/quantize-examples/efficient_quantize.py --model_path <模型路径> --quantize_type <量化类型> --output_dir <输出目录>

量化类型选项:
- dynamic_int8: PyTorch动态INT8量化（Windows完全兼容）
- dynamic_fp16: Float16半精度量化（Windows完全兼容）
- ggml_q4: GGML Q4量化（需要安装ctransformers库）
- ggml_q8: GGML Q8量化（需要安装ctransformers库）

注意: 本脚本不依赖bitsandbytes库，避免Windows兼容性问题
"""

import os
import sys
import time
import gc
import math
import argparse
import logging
import warnings
import platform
from pathlib import Path
import subprocess
import json

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(deepseek_dir)
sys.path.insert(0, project_root)

# 导入配置
from config.config import *

# 检查系统
if platform.system() != "Windows":
    print("注意：本脚本针对Windows系统优化，在其他系统上也可运行但可能不是最优方案")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quantize.log")
    ]
)
logger = logging.getLogger(__name__)

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*The model is in half precision but the dtype is not.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overriding torch_dtype=None with `torch_dtype=torch.float32`.*")

# 导入必要的库
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"缺少必要的依赖库: {e}")
    deps = ", ".join(QUANTIZE_DEPENDENCIES["basic"])
    logger.info(f"请先安装必要的依赖: pip install {deps}")
    sys.exit(1)

# 检查可选依赖
HAS_CTRANSFORMERS = False
try:
    from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
    HAS_CTRANSFORMERS = True
except ImportError:
    logger.warning("未安装ctransformers库，无法使用GGML量化方法")
    logger.info("可使用 pip install ctransformers 安装")

# 检查CUDA可用性
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    logger.info(f"检测到CUDA: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA版本: {torch.version.cuda}")
    logger.info(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    logger.warning("未检测到CUDA支持，将使用CPU进行处理（速度会很慢）")

def get_memory_info():
    """获取当前内存使用情况"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    info = {
        "RAM使用 (GB)": memory_info.rss / (1024 ** 3),
        "虚拟内存 (GB)": memory_info.vms / (1024 ** 3),
    }
    
    if HAS_CUDA:
        info["GPU已用内存 (GB)"] = torch.cuda.memory_allocated() / (1024 ** 3)
        info["GPU缓存内存 (GB)"] = torch.cuda.memory_reserved() / (1024 ** 3)
    
    return info

def log_memory_usage(stage):
    """记录内存使用情况"""
    mem_info = get_memory_info()
    logger.info(f"===== {stage} 内存使用情况 =====")
    for key, value in mem_info.items():
        logger.info(f"{key}: {value:.2f} GB")

def clean_memory():
    """清理内存"""
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        # 额外的内存清理（针对Windows）
        if platform.system() == "Windows":
            torch.cuda.synchronize()

def dynamic_int8_quantize(model):
    """应用动态INT8量化"""
    logger.info("开始应用PyTorch动态INT8量化...")
    
    def _quantize_module(module):
        """对模块进行递归量化"""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # 对线性层应用量化
                logger.debug(f"量化线性层: {name}")
                setattr(module, name, torch.quantization.quantize_dynamic(
                    child, {nn.Linear}, dtype=torch.qint8
                ))
                clean_memory()
            elif len(list(child.children())) > 0:
                # 递归处理非叶子模块
                _quantize_module(child)
        return module
    
    # 让模型处于评估模式
    model.eval()
    
    # 记录量化前的内存
    log_memory_usage("量化前")
    
    # 对模型的各个部分应用量化
    for name, module in tqdm(list(model.named_children()), desc="量化模型子模块"):
        try:
            if len(list(module.children())) > 0:
                # 只处理有子模块的部分
                logger.info(f"处理模块: {name}")
                
                # 量化前清理内存
                clean_memory()
                
                # 量化模块
                updated_module = _quantize_module(module)
                setattr(model, name, updated_module)
                
                # 量化后清理内存
                clean_memory()
        except Exception as e:
            logger.warning(f"量化模块 {name} 失败: {e}")
    
    logger.info("动态INT8量化完成")
    
    # 记录量化后的内存
    log_memory_usage("量化后")
    
    return model

def dynamic_fp16_quantize(model):
    """应用FP16半精度量化"""
    logger.info("开始应用FP16半精度量化...")
    if not HAS_CUDA:
        logger.warning("无GPU支持，FP16量化可能无法获得最佳性能")
    
    # 转换模型为FP16 (CPU上也可以使用，但没有性能优势)
    if HAS_CUDA:
        model = model.half().cuda()
    else:
        model = model.half()
    
    return model

def ggml_quantize(model_path, output_path, quantize_type="q4_k_m"):
    """使用GGML/GGUF格式进行量化"""
    if not HAS_CTRANSFORMERS:
        raise ImportError("GGML量化需要ctransformers库支持")
    
    logger.info(f"开始GGML量化，类型: {quantize_type}...")
    
    try:
        # 首先尝试使用huggingface-cli进行转换
        logger.info("尝试使用huggingface-cli进行转换...")
        
        # 检查是否安装了huggingface_hub[cli]
        try:
            subprocess.run("pip install -q huggingface_hub[cli]", shell=True, check=True)
        except subprocess.CalledProcessError:
            logger.warning("安装huggingface_hub[cli]失败")
        
        # 构建命令
        output_file = os.path.join(output_path, f"model_{quantize_type}.gguf")
        cmd = f'huggingface-cli convert-to-gguf {model_path} --outfile {output_file} --quantize {quantize_type}'
        
        # 执行命令
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode == 0:
            logger.info(f"GGML量化成功，已保存到: {output_file}")
            return output_file
        else:
            logger.warning("huggingface-cli转换失败，尝试替代方法...")
    except Exception as e:
        logger.warning(f"使用huggingface-cli量化失败: {e}")
    
    # 如果huggingface-cli失败，提供替代方案的说明
    logger.info("请考虑使用以下替代方法:")
    logger.info("1. 安装llama.cpp并手动转换: https://github.com/ggerganov/llama.cpp")
    logger.info("2. 使用quantize.exe工具（Windows版llama.cpp提供）")
    logger.info("3. 或使用其他量化方法，如PyTorch动态量化")
    
    return None

def quantize_model(args):
    """主量化函数"""
    model_path = args.model_path
    output_dir = args.output_dir
    quantize_type = args.quantize_type
    
    # 使用配置文件中的默认值
    if not output_dir:
        output_dir = QUANTIZE_OUTPUT_DIR
    
    if not quantize_type:
        quantize_type = DEFAULT_QUANTIZE_TYPE
    
    # 如果模型路径是模型ID，转换为本地路径
    if "/" in model_path and not os.path.exists(model_path):
        model_path = get_default_download_dir(model_path)
    
    # 记录配置
    logger.info("==== 量化配置 ====")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"量化类型: {quantize_type} - {QUANTIZE_TYPES.get(quantize_type, '未知')}")
    logger.info(f"系统类型: {platform.system()} {platform.release()}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录量化开始前的内存使用
    log_memory_usage("开始量化前")
    
    # 保存配置
    config = {
        "original_model": model_path,
        "quantize_type": quantize_type,
        "system": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if HAS_CUDA else "N/A",
        "quantize_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 保存分词器（所有量化方法都需要）
    logger.info("加载和保存分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)
    
    # 根据量化类型执行不同的量化方法
    if quantize_type == "ggml_q4" or quantize_type == "ggml_q8":
        # GGML量化
        q_type = "q4_k_m" if quantize_type == "ggml_q4" else "q8_0"
        output_file = ggml_quantize(model_path, output_dir, q_type)
        if output_file:
            logger.info(f"GGML量化完成，模型保存在: {output_file}")
            return output_dir
        else:
            logger.error("GGML量化失败")
            return None
    
    # PyTorch原生量化方法需要先加载模型
    logger.info("加载原始模型...")
    try:
        # 始终在CPU上加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE
        )
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None
    
    # 记录模型加载后的内存
    log_memory_usage("模型加载后")
    
    # 应用选定的量化方法
    try:
        if quantize_type == "dynamic_int8":
            # 动态INT8量化
            quantized_model = dynamic_int8_quantize(model)
        elif quantize_type == "dynamic_fp16":
            # FP16半精度量化
            quantized_model = dynamic_fp16_quantize(model)
        else:
            logger.error(f"不支持的量化类型: {quantize_type}")
            return None
    except Exception as e:
        logger.error(f"量化失败: {e}")
        return None
    
    # 保存量化后的模型
    logger.info(f"保存量化后的模型到: {output_dir}"),
    try:
        # 确保模型在CPU上以便保存
        quantized_model = quantized_model.cpu()
        quantized_model.save_pretrained(output_dir)
        logger.info(f"量化模型保存成功: {output_dir}")
    except Exception as e:
        logger.error(f"保存量化模型失败: {e}")
        
        # 尝试使用torch.save作为备选
        try:
            logger.info("尝试使用torch.save作为备选保存方法...")
            torch.save(quantized_model.state_dict(), os.path.join(output_dir, "model.pt"))
            logger.info(f"模型状态已保存到: {os.path.join(output_dir, 'model.pt')}")
            
            # 保存模型配置
            model.config.save_pretrained(output_dir)
            logger.info("模型配置已保存")
        except Exception as e2:
            logger.error(f"备选保存方法也失败: {e2}")
            return None
    
    # 最终清理内存
    del model
    del quantized_model
    clean_memory()
    
    logger.info("量化处理完成")
    return output_dir

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高效内存管理的模型量化工具")
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径（本地路径或HuggingFace模型ID）")
    parser.add_argument("--output_dir", type=str, default=QUANTIZE_OUTPUT_DIR, help="量化后模型的保存目录")
    parser.add_argument("--quantize_type", type=str, default=DEFAULT_QUANTIZE_TYPE, 
                      choices=list(QUANTIZE_TYPES.keys()),
                      help="量化类型")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    logger.info("=== 高效内存管理的模型量化工具 ===")
    start_time = time.time()
    
    output_path = quantize_model(args)
    
    if output_path:
        logger.info(f"量化处理成功完成，用时: {time.time() - start_time:.2f}秒")
        logger.info(f"量化后的模型保存在: {output_path}")
        
        # 写入说明文件
        with open(os.path.join(output_path, "README.txt"), "w") as f:
            f.write(f"""量化模型说明

原始模型: {args.model_path}
量化类型: {args.quantize_type} - {QUANTIZE_TYPES.get(args.quantize_type, '未知')}
量化时间: {time.strftime("%Y-%m-%d %H:%M:%S")}

如何使用量化后的模型:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("{output_path}")
model = AutoModelForCausalLM.from_pretrained("{output_path}")

# 使用示例
text = "请介绍一下人工智能:"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

注意: 本模型已使用 {args.quantize_type} 方法量化，如需查看原始模型性能对比，请使用 model_benchmark.py 脚本。
""")
    else:
        logger.error("量化处理失败") 