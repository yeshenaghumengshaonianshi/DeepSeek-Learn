"""
DeepSeek模型PyTorch原生量化示例

本示例展示如何使用PyTorch内置的量化功能对DeepSeek模型进行量化，
这种方法不依赖于bitsandbytes，在Windows系统上完全兼容。

使用方法: python deepSeek-learn/quantize-examples/torch_quantize_example.py
"""

import os
import sys
import time

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(deepseek_dir)
sys.path.insert(0, project_root)

# 导入配置和必要的库
from config.config import *

import torch
import torch.ao.quantization as quant
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def measure_performance(model, tokenizer, input_text, max_new_tokens=100):
    """测量模型推理性能"""
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end_time = time.time()
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "text": generated_text,
        "inference_time": end_time - start_time,
        "tokens_per_second": max_new_tokens / (end_time - start_time)
    }

def quantize_module(module, dtype=torch.qint8):
    """对模型的指定模块进行静态量化"""
    if isinstance(module, nn.Linear):
        # 对线性层进行量化
        return torch.quantization.quantize_dynamic(
            module, 
            {nn.Linear}, 
            dtype=dtype
        )
    else:
        # 递归处理子模块
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, torch.quantization.quantize_dynamic(
                    child, {nn.Linear}, dtype=dtype
                ))
            else:
                quantize_module(child, dtype)
    return module

def main():
    # 获取模型路径
    model_path = get_default_download_dir(DEFAULT_MODEL_NAME)
    print(f"加载模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 测试输入文本
    test_input = "请给我介绍一下量子计算的基本原理："
    
    # 1. 首先加载FP16模型进行基准测试
    # print("\n加载FP16模型（基准模型）...")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16_model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.float16,
    #     device_map=device
    # )
    
    # print("\n运行FP16模型基准测试...")
    # fp16_results = measure_performance(fp16_model, tokenizer, test_input)
    
    # # 内存占用
    # if torch.cuda.is_available():
    #     gpu_memory_fp16 = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
    #     print(f"FP16模型占用GPU内存: {gpu_memory_fp16:.2f} GB")
    
    # # 删除FP16模型释放内存
    # del fp16_model
    # torch.cuda.empty_cache()
    
    # 2. 使用PyTorch原生量化
    print("\n加载模型并使用PyTorch动态量化...")
    
    # 先加载FP32模型到CPU，然后进行量化
    print("加载FP32模型到CPU...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"  # 量化需要在CPU上进行
    )
    
    # 对模型的各部分进行量化
    print("正在量化模型...")
    model_quantized = model_fp32
    
    # 对transformers层进行量化
    for name, module in model_quantized.named_children():
        if "transformer" in name or "model" in name:
            print(f"量化 {name} 模块...")
            setattr(model_quantized, name, quantize_module(module))
    
    # 将量化后的模型移到可用设备上
    # if torch.cuda.is_available():
    #     model_quantized = model_quantized.to('cuda:0', non_blocking=True)
    #     torch.cuda.empty_cache()  # 清理缓存
    # else:
        model_quantized = model_quantized.to('cpu')
    
    # 3. 测试量化后的模型性能
    print("\n运行量化模型测试...")
    quant_results = measure_performance(model_quantized, tokenizer, test_input)
    
    # 内存占用
    if torch.cuda.is_available():
        gpu_memory_quant = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
        print(f"量化模型占用GPU内存: {gpu_memory_quant:.2f} GB")
        # print(f"内存减少: {(1 - gpu_memory_quant/gpu_memory_fp16) * 100:.2f}%")
    
    # 4. 输出比较结果
    print("\n====== 性能比较 ======")
    # print(f"FP16模型推理时间: {fp16_results['inference_time']:.2f}秒")
    print(f"量化模型推理时间: {quant_results['inference_time']:.2f}秒")
    # print(f"速度比例: {fp16_results['inference_time']/quant_results['inference_time']:.2f}")
    
    print("\n====== 输出质量对比 ======")
    # print("FP16模型输出:")
    # print(fp16_results['text'])
    print("\n量化模型输出:")
    print(quant_results['text'])
    
    # 5. 保存量化模型
    save_model = True
    if save_model:
        # 确保models目录存在
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # 设置保存路径
        save_path = os.path.join(models_dir, "torch_quantized_deepseek")
        
        print(f"\n保存量化模型到: {save_path}")
        try:
            model_quantized.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"量化模型已保存到: {save_path}")
            
            # 保存一个使用示例脚本
            example_script = os.path.join(save_path, "example_usage.py")
            with open(example_script, "w") as f:
                f.write("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载量化模型
model_path = "./torch_quantized_deepseek"  # 根据实际路径调整
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 测试输入
text = "请介绍一下人工智能:"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 生成回复
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
""")
            print(f"使用示例已保存到: {example_script}")
            
        except Exception as e:
            print(f"保存模型失败: {e}")
            print("使用JIT保存...")
            
            # 尝试使用TorchScript保存
            try:
                scripted_model = torch.jit.script(model_quantized)
                torch.jit.save(scripted_model, os.path.join(save_path, "model.pt"))
                print(f"模型已使用TorchScript保存到: {save_path}/model.pt")
            except Exception as e:
                print(f"使用TorchScript保存失败: {e}")
                print("请尝试其他保存方法")

if __name__ == "__main__":
    main() 