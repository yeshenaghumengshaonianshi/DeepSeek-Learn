"""
API服务器入口点，提供HTTP接口
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Optional, Union, Any

# 添加项目根目录到路径
file_dir = os.path.dirname(os.path.abspath(__file__))
vllm_dir = os.path.dirname(os.path.dirname(file_dir))
sys.path.insert(0, vllm_dir)

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.config import CONFIG

# 创建FastAPI应用
app = FastAPI(title="vLLM API服务器")

# 全局引擎实例
engine = None


@app.get("/")
async def root():
    """API根路径"""
    return {"status": "运行中", "version": CONFIG.VERSION, "model": engine.model_config.model_path if engine else "未加载"}


@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "健康"}


@app.post("/generate")
async def generate(request: Request):
    """生成文本接口
    
    请求格式:
    {
        "prompt": "你好，请介绍一下自己",
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    }
    """
    global engine
    
    # 解析请求JSON
    request_dict = await request.json()
    
    # 提取参数
    prompt = request_dict.pop("prompt")
    sampling_params_dict = request_dict.pop("sampling_params", {})
    
    # 创建采样参数
    sampling_params = SamplingParams(**sampling_params_dict)
    
    # 生成请求ID
    request_id = random_uuid()
    
    # 提交任务给引擎
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    # 获取结果
    try:
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # 客户端已断开连接
                break
            final_output = request_output
        
        if final_output is None:
            return JSONResponse({"error": "生成过程中出现错误"}, status_code=500)
        
        # 提取回复文本
        response = {
            "text": final_output.outputs[0].text,
            "finished": final_output.finished,
            "tokens_generated": len(final_output.outputs[0].token_ids),
        }
        
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/generate_stream")
async def generate_stream(request: Request):
    """流式生成文本接口"""
    global engine
    
    # 解析请求JSON
    request_dict = await request.json()
    
    # 提取参数
    prompt = request_dict.pop("prompt")
    sampling_params_dict = request_dict.pop("sampling_params", {})
    
    # 创建采样参数
    sampling_params = SamplingParams(**sampling_params_dict)
    
    # 生成请求ID
    request_id = random_uuid()

    # 定义流式响应函数
    async def response_stream():
        try:
            async for request_output in engine.generate(prompt, sampling_params, request_id):
                if await request.is_disconnected():
                    break
                
                # 构建输出
                output = {
                    "text": request_output.outputs[0].text,
                    "finished": request_output.finished,
                }
                
                # 转换为JSON行格式
                yield json.dumps(output) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
    
    # 返回流式响应
    return StreamingResponse(response_stream(), media_type="application/json")


@app.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "default_model": CONFIG.MODEL.DEFAULT_MODEL_PATH,
        "available_models": CONFIG.MODEL.AVAILABLE_MODELS
    }


def serve(
    model: str = CONFIG.MODEL.DEFAULT_MODEL_PATH,
    tensor_parallel_size: int = CONFIG.INFERENCE.DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = CONFIG.INFERENCE.DEFAULT_GPU_MEMORY_UTILIZATION,
    host: str = CONFIG.API.DEFAULT_HOST,
    port: int = CONFIG.API.DEFAULT_PORT,
    **kwargs,
):
    """启动API服务器
    
    Args:
        model: 模型名称或路径
        tensor_parallel_size: 用于张量并行的GPU数量
        gpu_memory_utilization: GPU内存利用率
        host: 服务器主机名
        port: 服务器端口
        **kwargs: 其他引擎参数
    """
    global engine
    
    # 解析模型路径
    model_path = CONFIG.MODEL.get_model_path(model)
    print(f"启动API服务器: {host}:{port}")
    print(f"加载模型: {model_path}")
    
    # 创建引擎参数
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs,
    )
    
    # 初始化引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 启动服务器
    uvicorn.run(app, host=host, port=port)


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description="vLLM API服务器")
    parser.add_argument("--model", type=str, default=CONFIG.MODEL.DEFAULT_MODEL_PATH, 
                        help=f"模型名称或路径 (默认: {CONFIG.MODEL.DEFAULT_MODEL_PATH})")
    parser.add_argument("--tensor-parallel-size", type=int, default=CONFIG.INFERENCE.DEFAULT_TENSOR_PARALLEL_SIZE, 
                        help=f"用于张量并行的GPU数量 (默认: {CONFIG.INFERENCE.DEFAULT_TENSOR_PARALLEL_SIZE})")
    parser.add_argument("--gpu-memory-utilization", type=float, default=CONFIG.INFERENCE.DEFAULT_GPU_MEMORY_UTILIZATION, 
                        help=f"GPU内存利用率 (默认: {CONFIG.INFERENCE.DEFAULT_GPU_MEMORY_UTILIZATION})")
    parser.add_argument("--host", type=str, default=CONFIG.API.DEFAULT_HOST, 
                        help=f"服务器主机名 (默认: {CONFIG.API.DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=CONFIG.API.DEFAULT_PORT, 
                        help=f"服务器端口 (默认: {CONFIG.API.DEFAULT_PORT})")
    
    args = parser.parse_args()
    
    # 解析模型名称
    if args.model:
        args.model = CONFIG.MODEL.get_model_path(args.model)
    
    serve(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main() 