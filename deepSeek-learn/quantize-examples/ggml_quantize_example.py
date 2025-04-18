"""
DeepSeek模型GGML/GGUF量化示例

本示例展示如何使用llama.cpp将DeepSeek模型转换为GGML/GGUF格式并进行量化。
这种方法在Windows系统上更可靠，不需要bitsandbytes库。

使用方法:
1. 安装依赖: pip install ctransformers
2. 运行: python deepSeek-learn/quantize-examples/ggml_quantize_example.py
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(deepseek_dir)
sys.path.insert(0, project_root)

# 导入配置
from config.config import *
import torch
from transformers import AutoTokenizer
from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM

def convert_to_gguf(model_path, output_path):
    """使用huggingface-cli将HF模型转换为GGUF格式"""
    try:
        print(f"开始将模型转换为GGUF格式...")
        # 检查是否已经安装huggingface-cli
        result = subprocess.run("pip install -q huggingface_hub[cli]", shell=True)
        if result.returncode != 0:
            raise Exception("安装huggingface-cli失败")
        
        # 使用huggingface-cli将模型转换为GGUF格式
        # q4_k_m 表示4位量化，使用k-means聚类算法
        cmd = f'huggingface-cli convert-to-gguf {model_path} --outfile {output_path} --quantize q4_k_m'
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode != 0:
            print("使用huggingface-cli转换失败，尝试使用ctransformers")
            raise Exception("使用huggingface-cli转换失败")
            
        print(f"模型已成功转换为GGUF格式并保存到: {output_path}")
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        print("请尝试手动安装llama.cpp并使用其进行转换")
        return False

def main():
    # 获取模型路径
    model_path = get_default_download_dir(DEFAULT_MODEL_NAME)
    print(f"原始模型路径: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 设置量化后模型保存路径
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # GGUF格式保存路径
    gguf_dir = os.path.join(models_dir, "gguf_quantized_deepseek")
    os.makedirs(gguf_dir, exist_ok=True)
    
    gguf_model_path = os.path.join(gguf_dir, "deepseek_q4km.gguf")
    
    # 1. 转换并量化模型为GGUF格式
    if not os.path.exists(gguf_model_path):
        success = convert_to_gguf(model_path, gguf_model_path)
        if not success:
            print("尝试使用替代方法...")
            try:
                # 保存分词器（ctransformers需要）
                tokenizer.save_pretrained(gguf_dir)
                print(f"分词器已保存到: {gguf_dir}")
                print("请手动使用llama.cpp转换模型并将结果保存到上述路径")
            except Exception as e:
                print(f"保存分词器失败: {e}")
                print("请尝试其他量化方法")
    else:
        print(f"已存在GGUF模型: {gguf_model_path}")
    
    # 2. 测试加载量化后的模型
    if os.path.exists(gguf_model_path):
        print("\n测试加载GGUF模型...")
        try:
            # 使用ctransformers加载GGUF模型
            model = CTAutoModelForCausalLM.from_pretrained(
                gguf_model_path,
                model_type="deepseek"
            )
            
            # 执行推理
            test_input = "请介绍一下量子计算的基本原理："
            print(f"\n输入: {test_input}")
            
            start_time = time.time()
            output = model.generate(test_input, max_new_tokens=100)
            end_time = time.time()
            
            print(f"\n输出: {output}")
            print(f"推理耗时: {end_time - start_time:.2f}秒")
            print("量化模型加载并测试成功！")
        except Exception as e:
            print(f"测试GGUF模型失败: {e}")
            print("请考虑使用其他量化库或方法")

if __name__ == "__main__":
    main() 