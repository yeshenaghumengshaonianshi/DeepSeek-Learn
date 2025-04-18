"""
DeepSeek模型混合CPU-GPU量化示例

本脚本实现了混合CPU-GPU量化方法，通过分层处理来克服GPU内存限制。
适用于Windows环境下GPU内存不足的情况。

使用方法: python deepSeek-learn/quantize-examples/hybrid_quantize.py
"""

import os
import sys
import time
import gc
import math

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(deepseek_dir)
sys.path.insert(0, project_root)

# 导入配置
from config.config import *

import torch
import torch.nn as nn
import torch.quantization as quant
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def get_device_info():
    """获取设备信息并返回可用的GPU内存"""
    if not torch.cuda.is_available():
        print("未检测到可用的GPU，将使用CPU进行全部操作")
        return None, 0
    
    device = torch.device("cuda:0")
    gpu_properties = torch.cuda.get_device_properties(device)
    total_memory = gpu_properties.total_memory / (1024**3)  # 转换为GB
    
    # 获取当前已使用和空闲内存
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
    free_memory = total_memory - allocated_memory - reserved_memory
    
    print(f"GPU: {gpu_properties.name}")
    print(f"总内存: {total_memory:.2f} GB")
    print(f"已分配内存: {allocated_memory:.2f} GB")
    print(f"空闲内存: {free_memory:.2f} GB")
    
    return device, free_memory

def quantize_layer_on_device(layer, device):
    """在指定设备上量化单个层"""
    if not device or device.type == "cpu":
        # 在CPU上进行动态量化
        if isinstance(layer, nn.Linear):
            return torch.quantization.quantize_dynamic(
                layer, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
    else:
        # 在GPU上量化可能需要特殊处理
        # 先将层移动到GPU
        try:
            layer = layer.to(device)
            # 执行一些GPU特定的量化（如果适用）
            # 这里可以根据实际情况增加GPU特定的量化代码
            # 然后将结果移回CPU
            layer = layer.cpu()
        except Exception as e:
            print(f"GPU量化失败，回退到CPU: {e}")
            if isinstance(layer, nn.Linear):
                return torch.quantization.quantize_dynamic(
                    layer, 
                    {nn.Linear}, 
                    dtype=torch.qint8
                )
    return layer

def hybrid_quantize_model(model, device=None, free_memory=0):
    """混合CPU-GPU量化模型"""
    # 确定哪些层在GPU上量化，哪些在CPU上量化
    gpu_layers_threshold = 100 * 1024 * 1024  # 参数数量阈值，小于此值的层尝试在GPU上量化
    
    if device and device.type == "cuda" and free_memory > 2.0:  # 至少有2GB可用内存
        print(f"使用混合CPU-GPU量化，GPU可用内存: {free_memory:.2f} GB")
    else:
        print("内存不足或无GPU，将使用纯CPU量化")
        device = None
    
    # 递归处理模型的每个层
    for name, module in tqdm(list(model.named_children()), desc="量化模型层"):
        # 检查是否为叶子模块
        if len(list(module.children())) == 0:
            # 叶子模块直接量化
            if isinstance(module, nn.Linear):
                num_params = sum(p.numel() for p in module.parameters())
                if device and num_params < gpu_layers_threshold:
                    # 小型层尝试在GPU上量化
                    try:
                        quantized_module = quantize_layer_on_device(module, device)
                        setattr(model, name, quantized_module)
                    except Exception as e:
                        print(f"GPU量化失败，回退到CPU: {e}")
                        quantized_module = quantize_layer_on_device(module, None)  # 使用CPU
                        setattr(model, name, quantized_module)
                else:
                    # 大型层在CPU上量化
                    quantized_module = quantize_layer_on_device(module, None)  # 使用CPU
                    setattr(model, name, quantized_module)
                
                # 每处理一些层后清理GPU内存
                if device and device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
        else:
            # 非叶子模块，递归处理
            hybrid_quantize_model(module, device, free_memory)
    
    return model

def optimize_for_inference(model):
    """为推理优化模型，减少内存使用"""
    # 模型结构特定优化可以在这里添加
    # 例如针对DeepSeek模型的特定优化
    
    # 设置为评估模式
    model.eval()
    
    # 冻结参数，避免梯度计算
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def quantize_and_save_model(model_path, save_path):
    """量化并保存模型"""
    print(f"开始加载原始模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 获取设备信息
    device, free_memory = get_device_info()
    
    # 清理GPU内存
    if device and device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        print("正在加载模型到CPU...")
        # 总是在CPU上加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        # 优化模型结构
        model = optimize_for_inference(model)
        
        # 执行混合量化
        print("开始混合CPU-GPU量化...")
        model = hybrid_quantize_model(model, device, free_memory)
        
        # 确保最终模型在CPU上
        model = model.cpu()
        
        # 保存量化后的模型
        print(f"保存量化模型到: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"混合量化完成。模型已保存到: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"模型量化失败: {e}")
        return None

def test_model(model_path, test_input="请解释一下量子计算的基本原理:"):
    """测试量化后的模型"""
    print(f"\n测试模型: {model_path}")
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 尝试确定可用内存并选择合适的设备
        if torch.cuda.is_available():
            # 检查GPU内存
            free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) - torch.cuda.memory_allocated(0) / (1024**3)
            if free_memory > 2.0:  # 如果有足够内存，使用GPU进行推理
                device = "cuda:0"
                print(f"推理使用GPU，可用内存: {free_memory:.2f} GB")
            else:
                device = "cpu"
                print("GPU内存不足，回退到CPU进行推理")
        else:
            device = "cpu"
            print("未检测到GPU，使用CPU进行推理")
        
        print(f"使用设备: {device}")
        
        # 加载模型
        try:
            # 首先尝试加载到选定设备
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device
            )
        except Exception as e:
            print(f"加载到{device}失败: {e}")
            print("回退到CPU加载")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu"
            )
            device = "cpu"
        
        # 进行模型推理
        print(f"测试输入: '{test_input}'")
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
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
    save_dir = os.path.join(models_dir, "hybrid_quantized")
    
    # 提取模型名称
    model_name = model_path.split('/')[-1]
    save_path = os.path.join(save_dir, f"hybrid_quantized_{model_name}")
    
    # 量化模型
    quantized_path = quantize_and_save_model(model_path, save_path)
    
    # 测试量化后的模型
    if quantized_path:
        test_model(quantized_path)

if __name__ == "__main__":
    main() 