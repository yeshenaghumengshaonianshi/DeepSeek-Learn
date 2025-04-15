import os
import sys

# 获取项目根目录并添加到模块搜索路径中
# 这样才能正确导入顶级包
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from config.config import (
    HF_ENDPOINT, 
    HF_HUB_DISABLE_SYMLINKS_WARNING,
    HF_HUB_ENABLE_HF_TRANSFER,
    TRANSFORMERS_OFFLINE,
    DEFAULT_MODEL_NAME,
    get_default_download_dir,
    MODEL_TRUST_REMOTE_CODE,
    MODEL_USE_FAST_TOKENIZER,
    MODEL_LOW_CPU_MEM_USAGE,
    MODEL_DEVICE_MAP
)

# 设置环境变量 - 必须在导入transformers之前设置
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = HF_HUB_DISABLE_SYMLINKS_WARNING
# 禁用 hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = HF_HUB_ENABLE_HF_TRANSFER
os.environ["TRANSFORMERS_OFFLINE"] = TRANSFORMERS_OFFLINE

# 导入必要的库
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# 使用前先执行
# pip install transformers==4.38.0 accelerate sentencepiece safetensors

def download_model_from_mirror(model_name: str, output_dir: str):
    """从指定的 Hugging Face 镜像站点下载模型和分词器。"""
    try:
        print(f"正在下载模型 {model_name}...")
        
        # 确保下载目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"模型将被下载到: {output_dir}")

        # 下载模型文件到指定目录
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"模型文件已下载到: {local_dir}")

        # 从本地加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            trust_remote_code=MODEL_TRUST_REMOTE_CODE,
            use_fast=MODEL_USE_FAST_TOKENIZER
        )
        print("分词器加载完成！")

        # 创建offload文件夹
        offload_folder = os.path.join(os.path.dirname(output_dir), "offload_folder")
        os.makedirs(offload_folder, exist_ok=True)
        
        # 加载模型时指定offload_folder
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            trust_remote_code=MODEL_TRUST_REMOTE_CODE,
            device_map=MODEL_DEVICE_MAP,
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
            offload_folder=offload_folder  # 添加卸载文件夹
        )
        print("模型加载完成！")

        return tokenizer, model

    except Exception as e:
        print(f"下载或加载失败：{e}")
        return None, None

def test_model_streaming(tokenizer, model):
    """测试模型流式输出功能"""
    if tokenizer is None or model is None:
        print("模型或分词器不可用，无法进行流式输出测试")
        return
    
    print("\n正在测试模型流式输出功能...")
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
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer
        )
    
    print("\n流式输出测试完成")


if __name__ == "__main__":
    # 检查是否已存在init_project.py脚本
    init_script = os.path.join(script_dir, "init_project.py")
    if os.path.exists(init_script):
        # 执行初始化项目目录结构的脚本
        print("初始化项目目录结构...")
        import util.init_project
        util.init_project.create_directory_structure()
    
    # 使用配置中的默认模型名称
    model_name = DEFAULT_MODEL_NAME
    
    # 使用配置中的默认下载目录
    download_dir = get_default_download_dir(model_name)
    
    print(f"将模型下载到: {download_dir}")

    # 调用函数下载模型
    tokenizer, model = download_model_from_mirror(model_name, download_dir)

    # 如果需要保存到本地路径，可以指定缓存目录
    if tokenizer and model:
        print("模型和分词器已成功加载！")
        print(f"模型路径: {download_dir}")
        print("使用此模型时，请将模型路径设置为上述路径。")
        
        # 测试流式输出功能
        test_model_streaming(tokenizer, model)
