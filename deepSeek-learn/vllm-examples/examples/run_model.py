#!/usr/bin/env python3
"""
示例脚本：使用vLLM引擎运行大型语言模型进行高效推理
"""

import argparse
import json
import time
from typing import List, Dict, Any

import torch
import sys
import os

# 添加项目根目录到路径，以便导入vllm包
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from vllm import LLM, SamplingParams
from vllm.config import CONFIG

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用vLLM运行模型的示例")
    
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG.MODEL.DEFAULT_MODEL_PATH,
        help=f"要加载的模型名称或路径 (默认: {CONFIG.MODEL.DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=CONFIG.INFERENCE.DEFAULT_TENSOR_PARALLEL_SIZE,
        help=f"用于张量并行的GPU数量 (默认: {CONFIG.INFERENCE.DEFAULT_TENSOR_PARALLEL_SIZE})"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=CONFIG.INFERENCE.DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"目标GPU内存利用率 (默认: {CONFIG.INFERENCE.DEFAULT_GPU_MEMORY_UTILIZATION})"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=CONFIG.INFERENCE.DEFAULT_MAX_TOKENS,
        help=f"生成的最大token数 (默认: {CONFIG.INFERENCE.DEFAULT_MAX_TOKENS})"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=CONFIG.INFERENCE.DEFAULT_TEMPERATURE,
        help=f"采样温度 (默认: {CONFIG.INFERENCE.DEFAULT_TEMPERATURE})"
    )
    
    parser.add_argument(
        "--prompts",
        type=str,
        default="请简单介绍一下vLLM的特点和优势。",
        help="要处理的提示或JSON文件路径"
    )
    
    parser.add_argument(
        "--api-mode",
        action="store_true",
        help="启动API服务器模式"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=CONFIG.API.DEFAULT_HOST,
        help=f"API服务器主机名 (默认: {CONFIG.API.DEFAULT_HOST})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=CONFIG.API.DEFAULT_PORT,
        help=f"API服务器端口 (默认: {CONFIG.API.DEFAULT_PORT})"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 解析模型名称/路径
    args.model = CONFIG.MODEL.get_model_path(args.model)
    
    return args

def load_prompts(prompts_arg: str) -> List[str]:
    """加载提示，可以是单个字符串或JSON文件路径"""
    if prompts_arg.endswith(".json"):
        with open(prompts_arg, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return [prompts_arg]

def run_inference(model_name: str, prompts: List[str], sampling_params: SamplingParams, 
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85):
    """运行模型推理"""
    print(f"加载模型: {model_name}")
    print(f"张量并行大小: {tensor_parallel_size}")
    print(f"GPU内存利用率: {gpu_memory_utilization}")
    
    # 初始化LLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 生成输出
    outputs = llm.generate(prompts, sampling_params)
    
    # 计算总时间
    total_time = time.time() - start_time
    
    return outputs, total_time

def start_api_server(model_name: str, tensor_parallel_size: int = 1, 
                    gpu_memory_utilization: float = 0.85,
                    host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器"""
    from vllm.entrypoints.api_server import serve
    
    print(f"启动API服务器: {host}:{port}")
    print(f"使用模型: {model_name}")
    
    serve(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        host=host,
        port=port,
    )

def main():
    """主函数"""
    args = parse_arguments()
    
    print(f"vLLM 版本: {CONFIG.VERSION}")
    print(f"使用模型: {args.model}")
    
    if args.api_mode:
        print(f"启动API服务器模式")
        start_api_server(
            args.model, 
            args.tensor_parallel_size, 
            args.gpu_memory_utilization,
            args.host,
            args.port
        )
    else:
        # 加载提示
        prompts = load_prompts(args.prompts)
        print(f"加载了 {len(prompts)} 个提示")
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=CONFIG.INFERENCE.DEFAULT_TOP_P,
            top_k=CONFIG.INFERENCE.DEFAULT_TOP_K
        )
        
        # 运行推理
        outputs, total_time = run_inference(
            args.model,
            prompts,
            sampling_params,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
        )
        
        # 打印结果
        print("\n===== 生成结果 =====")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
            print(f"\n提示 {i+1}: {prompt}")
            print(f"生成: {generated_text}")
            
            # 计算性能指标
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            
            print(f"提示tokens: {prompt_tokens}")
            print(f"生成tokens: {completion_tokens}")
            print(f"生成速度: {tokens_per_second:.2f} tokens/sec")
        
        print(f"\n总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main() 