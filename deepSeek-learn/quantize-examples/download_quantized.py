"""
下载预量化的DeepSeek模型

本脚本从Hugging Face下载已经量化好的DeepSeek模型，
适合Windows环境下无法进行本地量化的情况。

使用方法: python deepSeek-learn/quantize-examples/download_quantized.py
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
from transformers import AutoTokenizer, AutoModelForCausalLM

# 预量化模型列表
QUANTIZED_MODELS = {
    "4bit": {
        # 4位量化DeepSeek模型
        "DeepSeek-7B": "TheBloke/deepseek-llm-7b-base-GPTQ",
        "DeepSeek-Coder-7B": "TheBloke/deepseek-coder-6.7b-base-GPTQ",
        "DeepSeek-R1-Instruct-7B": "TheBloke/deepseek-llm-7b-chat-GPTQ"
    },
    "8bit": {
        # 8位量化DeepSeek模型 
        "DeepSeek-7B": "TheBloke/deepseek-llm-7b-base-AWQ",
        "DeepSeek-Coder-7B": "TheBloke/deepseek-coder-6.7b-base-AWQ",
        "DeepSeek-R1-Instruct-7B": "TheBloke/deepseek-llm-7b-chat-AWQ"
    }
}

def download_model(model_name, bits=4):
    """下载预量化模型"""
    # 获取对应位数的模型列表
    models = QUANTIZED_MODELS.get(f"{bits}bit", {})
    
    # 检查是否有对应的预量化模型
    if model_name not in models:
        similar_models = []
        for key in models.keys():
            if model_name.lower() in key.lower():
                similar_models.append(key)
        
        if similar_models:
            print(f"未找到完全匹配的预量化模型，但发现类似模型: {similar_models}")
            model_name = similar_models[0]
            print(f"使用: {model_name}")
        else:
            print(f"错误: 未找到匹配的{bits}位预量化模型，可用选项: {list(models.keys())}")
            return None
    
    # 获取模型ID
    model_id = models[model_name]
    print(f"开始下载预量化模型: {model_id}")
    
    # 设置保存路径
    models_dir = os.path.join(project_root, "models")
    save_path = os.path.join(models_dir, f"{bits}bit_quantized_{model_name.lower().replace('-', '_')}")
    
    try:
        # 加载模型和分词器
        print(f"从Hugging Face下载预量化模型: {model_id}")
        
        # 下载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 下载模型
        model_type = "GPTQ" if bits == 4 else "AWQ"
        
        if model_type == "GPTQ":
            try:
                from transformers import GPTQConfig
                
                # 创建GPTQ配置
                gptq_config = GPTQConfig(bits=4, disable_exllama=True)
                
                # 加载模型
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=gptq_config
                )
            except:
                print("尝试使用auto-gptq加载...")
                try:
                    from auto_gptq import AutoGPTQForCausalLM
                    
                    model = AutoGPTQForCausalLM.from_pretrained(
                        model_id,
                        device="cuda:0" if torch.cuda.is_available() else "cpu"
                    )
                except Exception as e:
                    print(f"GPTQ加载失败: {e}")
                    print("尝试普通方式加载...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto"
                    )
                    
        elif model_type == "AWQ":
            try:
                # 尝试加载AWQ模型
                from awq import AutoAWQForCausalLM
                
                model = AutoAWQForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto"
                )
            except:
                print("尝试使用普通方式加载AWQ模型...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto"
                )
        
        # 保存模型和分词器
        print(f"保存预量化模型到: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"模型保存成功: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"下载预量化模型失败: {e}")
        
        # 推荐备选方案
        print("\n备选方案:")
        print("1. 访问 https://huggingface.co/TheBloke 手动下载量化模型")
        print("2. 尝试使用其他量化方法 (如simple_cpu_quantize.py)")
        print("3. 检查网络连接或代理设置")
        
        return None

def test_model(model_path, test_input="请简单介绍一下量子计算:"):
    """测试下载的量化模型"""
    print(f"\n测试模型: {model_path}")
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 设置设备
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 尝试多种方式加载模型
        model = None
        try:
            # 尝试常规加载
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device if device == "cpu" else "auto"
            )
        except:
            try:
                # 尝试GPTQ加载
                from auto_gptq import AutoGPTQForCausalLM
                model = AutoGPTQForCausalLM.from_pretrained(model_path)
            except:
                try:
                    # 尝试AWQ加载
                    from awq import AutoAWQForCausalLM
                    model = AutoAWQForCausalLM.from_pretrained(model_path)
                except Exception as e:
                    print(f"所有加载方式均失败: {e}")
                    return False
        
        # 进行模型推理
        print(f"测试输入: '{test_input}'")
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
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
    # 从配置获取当前使用的模型名称
    model_name = DEFAULT_MODEL_NAME.split('/')[-1]  # 提取模型名称部分
    
    # 主要DeepSeek型号映射
    if "DeepSeek-7B" in model_name or "deepseek-llm-7b" in model_name.lower():
        model_type = "DeepSeek-7B"
    elif "Coder" in model_name or "coder" in model_name.lower():
        model_type = "DeepSeek-Coder-7B"
    elif "R1" in model_name or "instruct" in model_name.lower() or "chat" in model_name.lower():
        model_type = "DeepSeek-R1-Instruct-7B"
    else:
        model_type = "DeepSeek-7B"  # 默认
    
    print(f"检测到的模型类型: {model_type}")
    
    # 选择量化位数
    bits = 4  # 默认4位量化，可以改为8
    
    # 下载预量化模型
    model_path = download_model(model_type, bits)
    
    # 测试模型
    if model_path:
        test_model(model_path)

if __name__ == "__main__":
    main() 