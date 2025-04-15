"""
简单的DeepSeek模型对话示例

这个脚本演示如何加载已下载的DeepSeek模型并进行简单对话。
使用方法: python examples/simple_chat.py
"""
import os
import sys

# 获取项目根目录并添加到模块搜索路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 导入配置和必要的库
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

# 设置历史对话的最大长度，避免上下文太长
MAX_HISTORY_LENGTH = 5

class DeepSeekChat:
    """DeepSeek模型对话类"""
    
    def __init__(self, model_path=None):
        """
        初始化对话类
        
        Args:
            model_path: 模型路径，默认使用配置中的路径
        """
        # 如果没有指定模型路径，使用默认路径
        if model_path is None:
            model_path = get_default_download_dir(DEFAULT_MODEL_NAME)
            
        print(f"正在加载模型: {model_path}")
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径 {model_path} 不存在！请先下载模型。")
        
        # 创建offload文件夹，用于降低内存使用
        offload_folder = os.path.join(os.path.dirname(os.path.dirname(model_path)), "offload_folder")
        os.makedirs(offload_folder, exist_ok=True)
        
        # 加载分词器
        print("正在加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=MODEL_TRUST_REMOTE_CODE,
            use_fast=MODEL_USE_FAST_TOKENIZER
        )
        
        # 检查设备 - 优先使用GPU，如无GPU则使用CPU
        if torch.cuda.is_available():
            device_map = MODEL_DEVICE_MAP  # 自动选择最佳设备配置
            print("使用GPU进行推理")
        else:
            device_map = "cpu"
            print("未检测到GPU，使用CPU进行推理（速度较慢）")
        
        # 加载模型
        print("正在加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=MODEL_TRUST_REMOTE_CODE,
            torch_dtype=torch.float16,  # 使用半精度，减少内存使用
            device_map=device_map,      # 设备映射
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
            offload_folder=offload_folder
        )
        
        # 创建文本流式输出器，用于流式显示生成结果
        self.streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
        
        # 初始化对话历史
        self.history = []
        
        print("模型加载完成，可以开始对话了！")
    
    def build_prompt(self, user_input):
        """
        构建完整的对话提示，包含历史对话和当前输入
        
        Args:
            user_input: 用户当前的输入
            
        Returns:
            完整的提示文本
        """
        # 构建提示的基本格式
        prompt = ""
        
        # 添加历史对话
        for turn in self.history[-MAX_HISTORY_LENGTH:]:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        
        # 添加当前用户输入
        prompt += f"User: {user_input}\nAssistant: "
        
        return prompt
    
    def generate_response(self, user_input):
        """
        生成回复
        
        Args:
            user_input: 用户输入
            
        Returns:
            模型生成的回复文本
        """
        # 构建完整提示
        prompt = self.build_prompt(user_input)
        
        # 对提示进行分词处理
        inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        
        # 使用模型生成回复（流式输出）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2000,  # 生成的最大tokens数量
                do_sample=True,       # 使用采样生成
                temperature=0.7,      # 温度参数，控制随机性
                top_p=0.9,           # Top-p采样，控制词汇分布
                streamer=self.streamer # 流式输出
            )
        
        # 解码输出，得到完整的响应文本（包括提示）
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取出模型的回复部分（去除提示部分）
        assistant_response = full_response[len(prompt):]
        
        # 更新对话历史
        self.history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        return assistant_response
    
    def chat(self):
        """运行交互式对话循环"""
        print("\n欢迎使用DeepSeek对话模型！输入'exit'或'quit'退出对话。")
        print("正在等待您的输入...\n")
        
        while True:
            # 获取用户输入
            user_input = input("User: ").strip()
            
            # 检查是否退出
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("感谢使用，再见！")
                break
            
            # 生成并打印回复
            print("Assistant: ", end="", flush=True)
            response = self.generate_response(user_input)
            # 在流式输出后添加换行
            print()

def main():
    """主函数"""
    try:
        # 创建聊天实例
        chat_bot = DeepSeekChat()
        
        # 开始对话
        chat_bot.chat()
        
    except KeyboardInterrupt:
        print("\n对话被中断。再见！")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 