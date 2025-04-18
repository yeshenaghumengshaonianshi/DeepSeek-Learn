"""
DeepSeek模型简易CPU量化示例

这个脚本在CPU上进行模型量化，适用于GPU内存受限的Windows环境。
使用方法: python deepSeek-learn/quantize-examples/simple_cpu_quantize.py
"""

import os
import sys
import time

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(deepseek_dir)
sys.path.insert(0, project_root)

# 导入配置
from config.config import *

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def quantize_model_cpu(model_path, save_path, bits=8):
    """在CPU上进行模型量化，无需大量GPU内存"""
    print(f"开始加载原始模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 设置量化配置
    quantization_config = None
    if bits == 8:
        print("使用8位量化...")
        quantization_config = {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0
        }
    elif bits == 4:
        print("使用4位量化...")
        try:
            # 尝试使用4位量化配置，需要bitsandbytes库
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4"
            }
        except Exception as e:
            print(f"4位量化配置失败: {e}")
            print("回退到8位量化...")
            quantization_config = {
                "load_in_8bit": True
            }
    
    # 优先尝试CPU量化方法
    try:
        # 设置环境变量，强制使用CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        print("第1步: 加载模型并量化 (CPU方式)...")
        
        # 使用CPU加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        # 手动对线性层进行量化
        print("对线性层应用动态量化...")
        model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        # 保存量化后的模型
        print(f"保存量化模型到: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"CPU量化完成。模型已保存到: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"CPU量化失败: {e}")
        return None

def test_model(model_path, test_input="请解释一下量子计算的基本原理:"):
    """测试量化后的模型"""
    print(f"\n测试模型: {model_path}")
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 设置设备 - 优先使用CPU避免内存问题
        device = "cpu"  # 或者 "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 加载模型到CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu"
        )
        
        # 进行模型推理
        print(f"测试输入: '{test_input}'")
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
        end_time = time.time()
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"模型输出: '{response}'")
        print(f"推理耗时: {end_time - start_time:.2f}秒")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def main():
    # 从配置获取模型路径
    model_path = DEFAULT_MODEL_NAME
    print(f"使用模型: {model_path}")
    
    # 设置保存路径
    models_dir = os.path.join(project_root, "models")
    save_dir = os.path.join(models_dir, "quantized_cpu")
    
    # 提取模型名称
    model_name = model_path.split('/')[-1]
    save_path = os.path.join(save_dir, f"cpu_quantized_{model_name}")
    
    # 量化模型
    quantized_path = quantize_model_cpu(model_path, save_path)
    
    # 测试量化后的模型
    if quantized_path:
        test_model(quantized_path)

if __name__ == "__main__":
    main() 