"""
DeepSeek模型INT8量化示例

本示例展示如何对DeepSeek模型进行INT8量化，减小模型体积并加速推理。
使用方法: python deepSeek-learn/quantize-examples/ptq_int8_example.py
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

def main():
    # 获取模型路径
    model_path = get_default_download_dir(DEFAULT_MODEL_NAME)
    print(f"加载模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 测试输入文本
    test_input = "请给我介绍一下量子计算的基本原理："
    
    # 1. 首先加载FP16模型进行基准测试
    print("\n加载FP16模型（基准模型）...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=MODEL_DEVICE_MAP
    )
    
    print("\n运行FP16模型基准测试...")
    fp16_results = measure_performance(fp16_model, tokenizer, test_input)
    
    # 内存占用
    gpu_memory_fp16 = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
    print(f"FP16模型占用GPU内存: {gpu_memory_fp16:.2f} GB")
    
    # 删除FP16模型释放内存
    del fp16_model
    torch.cuda.empty_cache()
    
    # 2. 加载INT8量化模型
    print("\n加载INT8量化模型...")
    
    # 创建8位量化配置
    int8_config = BitsAndBytesConfig(
        load_in_8bit=True,              # 启用8位量化
        llm_int8_threshold=6.0,         # 量化阈值
        llm_int8_has_fp16_weight=False  # 不保留FP16权重
    )
    # 实现INT8量化
    int8_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=int8_config,
        device_map=MODEL_DEVICE_MAP
    )
    
    print("\n运行INT8模型测试...")
    int8_results = measure_performance(int8_model, tokenizer, test_input)
    
    # 内存占用
    gpu_memory_int8 = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
    print(f"INT8模型占用GPU内存: {gpu_memory_int8:.2f} GB")
    
    # 3. 输出比较结果
    print("\n====== 性能比较 ======")
    print(f"FP16模型推理时间: {fp16_results['inference_time']:.2f}秒")
    print(f"INT8模型推理时间: {int8_results['inference_time']:.2f}秒")
    print(f"速度提升: {fp16_results['inference_time']/int8_results['inference_time']:.2f}倍")
    print(f"内存减少: {(1 - gpu_memory_int8/gpu_memory_fp16) * 100:.2f}%")
    
    print("\n====== 输出质量对比 ======")
    print("FP16模型输出:")
    print(fp16_results['text'])
    print("\nINT8模型输出:")
    print(int8_results['text'])
    
    # 4. 保存量化模型
    save_model = True # 启用模型保存
    if save_model:
        # 确保models目录存在
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # 设置保存路径为项目根目录下的models/int8_quantized_deepseek7B
        save_path = os.path.join(models_dir, "int8_quantized_deepseek7B")
        
        print(f"\n保存INT8量化模型到: {save_path}")
        int8_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"量化模型已保存到: {save_path}")
        
        # 释放原始INT8模型内存
        del int8_model
        torch.cuda.empty_cache()
        
        # 5. 验证保存的量化模型
        print("\n======= 验证保存的量化模型 =======")
        print(f"从 {save_path} 加载量化模型...")
        
        try:
            # 尝试加载保存的模型
            saved_model = AutoModelForCausalLM.from_pretrained(
                save_path,
                device_map=MODEL_DEVICE_MAP
            )
            saved_tokenizer = AutoTokenizer.from_pretrained(save_path)
            
            # 使用加载的模型进行推理
            verification_input = "解释一下人工智能的伦理挑战："
            print(f"\n输入验证问题: {verification_input}")
            
            # 执行推理
            inputs = saved_tokenizer(verification_input, return_tensors="pt").to(saved_model.device)
            with torch.no_grad():
                outputs = saved_model.generate(**inputs, max_new_tokens=100)
            generated_text = saved_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("\n验证模型输出:")
            print(generated_text)
            print("\n模型验证成功! 量化模型可以正常使用。")
            
        except Exception as e:
            print(f"验证失败: {e}")
            print("请检查模型保存是否成功或尝试使用其他加载方式。")

if __name__ == "__main__":
    main() 