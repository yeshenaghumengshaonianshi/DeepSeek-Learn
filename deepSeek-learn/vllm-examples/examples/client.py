#!/usr/bin/env python3
"""
vLLM API客户端示例

演示如何调用vLLM API服务器进行文本生成
"""

import argparse
import json
import time
import sys
import os
import requests
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from vllm.config import CONFIG


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="vLLM API客户端示例")
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=f"http://{CONFIG.API.DEFAULT_HOST}:{CONFIG.API.DEFAULT_PORT}",
        help=f"vLLM API服务器URL (默认: http://{CONFIG.API.DEFAULT_HOST}:{CONFIG.API.DEFAULT_PORT})"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用流式API"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="请介绍一下vLLM的特点和优势。",
        help="提示文本"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=CONFIG.INFERENCE.DEFAULT_TEMPERATURE,
        help=f"采样温度 (默认: {CONFIG.INFERENCE.DEFAULT_TEMPERATURE})"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=CONFIG.INFERENCE.DEFAULT_MAX_TOKENS,
        help=f"生成的最大token数 (默认: {CONFIG.INFERENCE.DEFAULT_MAX_TOKENS})"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=CONFIG.INFERENCE.DEFAULT_TOP_P,
        help=f"nucleus采样概率 (默认: {CONFIG.INFERENCE.DEFAULT_TOP_P})"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出服务器上可用的模型"
    )
    
    return parser.parse_args()


def list_available_models(api_url: str):
    """获取服务器上可用的模型列表"""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/models")
        if response.status_code == 200:
            models = response.json()
            print("\n可用模型:")
            print(f"默认模型: {models['default_model']}")
            print("所有模型:")
            for name, path in models['available_models'].items():
                print(f" - {name}: {path}")
            return True
        else:
            print(f"获取模型列表失败: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"请求错误: {e}")
        return False


def generate_text(
    api_url: str,
    prompt: str,
    temperature: float = CONFIG.INFERENCE.DEFAULT_TEMPERATURE,
    max_tokens: int = CONFIG.INFERENCE.DEFAULT_MAX_TOKENS,
    top_p: float = CONFIG.INFERENCE.DEFAULT_TOP_P,
    stream: bool = False
):
    """从API生成文本
    
    Args:
        api_url: API服务器URL
        prompt: 提示文本
        temperature: 采样温度
        max_tokens: 生成的最大token数
        top_p: nucleus采样概率
        stream: 是否使用流式API
    
    Returns:
        生成的文本
    """
    # 准备请求数据
    payload = {
        "prompt": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
    }
    
    # 选择API端点
    endpoint = "/generate_stream" if stream else "/generate"
    api_endpoint = f"{api_url.rstrip('/')}{endpoint}"
    
    print(f"发送请求到: {api_endpoint}")
    print(f"提示: {prompt}")
    print("正在生成...")
    
    start_time = time.time()
    
    if stream:
        # 流式请求
        response = requests.post(api_endpoint, json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"错误: {response.status_code}")
            print(response.text)
            return None
        
        # 处理流式响应
        full_text = ""
        for line in response.iter_lines():
            if line:
                # 解析JSON行
                try:
                    result = json.loads(line)
                    if "error" in result:
                        print(f"错误: {result['error']}")
                        break
                    
                    text = result.get("text", "")
                    if text != full_text:
                        # 只打印新内容
                        print(text[len(full_text):], end="", flush=True)
                        full_text = text
                    
                    if result.get("finished", False):
                        print("\n")
                        break
                except json.JSONDecodeError:
                    print(f"无法解析: {line}")
        
        return full_text
    else:
        # 非流式请求
        response = requests.post(api_endpoint, json=payload)
        
        if response.status_code != 200:
            print(f"错误: {response.status_code}")
            print(response.text)
            return None
        
        result = response.json()
        generated_text = result.get("text", "")
        
        # 打印结果
        print("\n生成的文本:")
        print(generated_text)
        
        # 打印性能信息
        tokens_generated = result.get("tokens_generated", 0)
        total_time = time.time() - start_time
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        print(f"\n生成了 {tokens_generated} 个tokens")
        print(f"耗时: {total_time:.2f}秒")
        print(f"速度: {tokens_per_second:.2f} tokens/sec")
        
        return generated_text


def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查版本
    print(f"vLLM 客户端版本: {CONFIG.VERSION}")
    
    # 列出模型
    if args.list_models:
        if list_available_models(args.api_url):
            return
    
    # 检查API连接
    try:
        health_check = requests.get(f"{args.api_url}/health")
        if health_check.status_code != 200:
            print(f"警告: API服务器健康检查失败 ({health_check.status_code})")
    except requests.RequestException as e:
        print(f"错误: 无法连接到API服务器 - {e}")
        return
    
    # 生成文本
    generate_text(
        api_url=args.api_url,
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stream=args.stream
    )


if __name__ == "__main__":
    main() 