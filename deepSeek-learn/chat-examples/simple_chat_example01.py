"""
最简单的DeepSeek模型调用示例

这个脚本演示如何加载已下载的DeepSeek模型并进行一次简单推理。
使用方法: python deepSeek-learn/simple_inference.py
"""

import os
import sys

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取examples目录
examples_dir = os.path.dirname(script_dir)
# 获取项目根目录 (llm-LearnNote)
project_root = os.path.dirname(examples_dir)
# 将项目根目录添加到Python模块搜索路径，以便能够导入config模块
sys.path.insert(0, project_root)

# 导入配置和必要的库
from config.config import (
    get_default_download_dir,
    DEFAULT_MODEL_NAME,
    MODEL_TRUST_REMOTE_CODE,
    MODEL_DEVICE_MAP,
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# 创建自定义的TextStreamer子类，用于直接显示不包含提示词的输出
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, prompt, skip_prompt=True, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.skip_prompt = skip_prompt
        self.prompt_len = len(prompt)
        self.started = False
        self.prompt = prompt

    def on_finalized_text(self, text, stream_end=False):
        # 计算提示词的长度（仅在第一次调用时）
        if not self.started and self.skip_prompt:
            self.started = True

            # 仅输出提示之后的文本
            if len(text) > self.prompt_len:
                print(text[self.prompt_len:], end="", flush=True)
        elif self.skip_prompt:
            # 继续输出文本
            print(text, end="", flush=True)
        else:
            # 原始行为
            super().on_finalized_text(text, stream_end)


def simple_inference():
    """
    加载模型并进行一次简单推理
    """
    # 获取模型路径
    model_path = get_default_download_dir(DEFAULT_MODEL_NAME)

    print(f"加载模型: {model_path}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=MODEL_TRUST_REMOTE_CODE
    )

    # 加载模型
    # 加载模型这步骤确定是在cpu还是gpu执行
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=MODEL_TRUST_REMOTE_CODE, 
        torch_dtype=torch.float16,
        # device_map=MODEL_DEVICE_MAP,  # 明确指定设备映射
        device_map=MODEL_DEVICE_MAP,
    )

    # 设置简单提示
    prompt = "请介绍一下自己："

    print(f"\n输入: {prompt}")

    # 对提示进行分词处理
    # 将输入文本转换为数字序列，并确保它与模型在同一设备上，这是深度学习模型处理文本的必要步骤。
    # pt 表示返回的类型为 pytorch 的 tensor(张量) 如果没有指定这个参数，默认会返回普通的Python列表
    # to方法是PyTorch张量的方法，用于将张量移动到特定设备上
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    # 创建流式输出处理器，自动跳过提示词
    streamer = CustomStreamer(tokenizer, prompt=prompt, skip_prompt=True, skip_special_tokens=True)
    
    print("\n输出: ", end="", flush=True)
    
    # 使用模型生成回复
    # with torch.no_grad()作用临时禁用梯度计算，提高推理速度并减少内存使用
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=model.config.eos_token_id,
            max_new_tokens=100,  # 生成的最大tokens数量
            temperature=0.7,      # 温度参数，控制随机性 值越低分布越集中,值越高分布越分散
            top_p=0.1,           # Top-p采样，控制词汇分布 值越小结果越稳定,值越大结果越多样
            # do_sample=True,  # 启用采样以获得更多样化的输出
            # streamer=TextStreamer(tokenizer, skip_special_tokens=True),
            streamer=streamer,  # 使用自定义流式输出处理器
        )

    # 解码输出，得到响应文本（不再需要打印，因为streamer已经处理了）
    print("\n")  # 添加最终的换行


if __name__ == "__main__":
    simple_inference()
