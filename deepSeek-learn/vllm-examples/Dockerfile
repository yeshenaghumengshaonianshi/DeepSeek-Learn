FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 创建模型目录
RUN mkdir -p /app/models/deepseek-ai

# 复制项目文件
COPY . /app/

# 安装vLLM依赖和包
RUN pip3 install --no-cache-dir -e .

# 安装额外的依赖
RUN pip3 install --no-cache-dir \
    torch>=2.0.0 \
    accelerate \
    transformers>=4.33.0 \
    sentencepiece

# 设置环境变量
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

# 暴露API端口
EXPOSE 8000

# 启动API服务器
CMD ["python3", "examples/run_model.py", "--api-mode", "--host", "0.0.0.0"]

# 使用说明
# 1. 构建镜像: docker build -t vllm-demo .
# 2. 启动容器: docker run --gpus all -p 8000:8000 -v /path/to/models:/app/models vllm-demo
# 3. 测试API: curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "你好，请介绍一下自己", "sampling_params": {"temperature": 0.7, "max_tokens": 100}}' 