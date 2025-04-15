import os
import sys

# 获取项目根目录并添加到模块搜索路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from config.config import (
    get_default_download_dir,
    DEFAULT_MODEL_NAME,
    MODEL_TRUST_REMOTE_CODE,
    MODEL_USE_FAST_TOKENIZER,
    MODEL_LOW_CPU_MEM_USAGE,
    MODEL_DEVICE_MAP
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

def verify_model(model_path):
    """验证模型是否可以正确加载"""
    print(f"正在验证模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径 {model_path} 不存在!")
        return False
    
    # 检查是否有必要的文件
    required_files = ["config.json", "tokenizer.json"]
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            print(f"错误: 关键文件 {file} 不存在!")
            return False
    
    # 创建offload文件夹
    offload_folder = os.path.join(os.path.dirname(os.path.dirname(model_path)), "offload_folder")
    os.makedirs(offload_folder, exist_ok=True)
    
    try:
        # 尝试加载tokenizer
        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=MODEL_TRUST_REMOTE_CODE,
            use_fast=MODEL_USE_FAST_TOKENIZER
        )
        print("分词器加载成功!")
        
        # 检查可用显存
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            print(f"可用GPU显存: {free_memory_gb:.2f} GB")
            
            if free_memory_gb < 6:
                print(f"警告: 可用显存不足 6GB，模型可能无法完全加载到GPU。将使用CPU+GPU混合模式。")
                device_map = MODEL_DEVICE_MAP
            else:
                device_map = MODEL_DEVICE_MAP
        else:
            print("未检测到GPU，将使用CPU加载模型")
            device_map = "cpu"
        
        # 尝试加载模型
        print("正在加载模型...")
        try:
            # 先尝试使用safetensors加载
            print("尝试使用safetensors加载...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=MODEL_TRUST_REMOTE_CODE,
                torch_dtype=torch.float16,
                device_map=device_map,
                low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
                offload_folder=offload_folder
            )
        except Exception as e:
            print(f"使用safetensors加载失败: {e}")
            print("尝试备用加载方式...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=MODEL_TRUST_REMOTE_CODE,
                torch_dtype=torch.float16,
                device_map=device_map,
                low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
                offload_folder=offload_folder,
                offload_state_dict=True  # 如果显存不足，将状态字典卸载到CPU
            )
        
        print("模型加载成功!")
        
        # 尝试简单推理 - 使用流式输出
        print("\n正在进行简单推理测试...")
        input_text = "你好，请介绍一下你自己"
        inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)
        
        print(f"输入: {input_text}")
        print(f"输出: ", end="", flush=True)
        
        # 创建文本流式输出器
        streamer = TextStreamer(tokenizer, skip_special_tokens=True)
        
        with torch.no_grad():
            # 使用streamer参数启用流式输出
            _ = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer
            )
        # 流式输出完成后添加换行
        print()  
        return True
    
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False

if __name__ == "__main__":
    # 首先初始化项目目录结构
    init_script = os.path.join(script_dir, "init_project.py")
    if os.path.exists(init_script):
        print("初始化项目目录结构...")
        import util.init_project
        util.init_project.create_directory_structure()
        
    # 获取模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 使用配置中的默认模型路径
        model_path = get_default_download_dir(DEFAULT_MODEL_NAME)
    
    # 验证模型
    success = verify_model(model_path)
    
    if success:
        print("\n模型验证成功! 可以正常使用该模型。")
    else:
        print("\n模型验证失败! 请检查模型文件或尝试重新下载。")
        print("您也可以尝试以下解决方案:")
        print("1. 确保已安装 safetensors 库: pip install safetensors")
        print("2. 确保有足够的显存或系统内存")
        print("3. 尝试使用规模较小的模型")