# LLM-LearnNote

这个仓库包含大型语言模型(LLM)学习和应用的示例代码、工具和笔记。

## 项目结构

```
LLM-LearnNote/
├── deepSeek-learn/           # DeepSeek模型相关代码
│   ├── chat-examples/        # 聊天应用示例
│   ├── finetune-examples/    # 微调示例
│   ├── quantize-examples/    # 量化示例
│   │   ├── data/             # 测试数据
│   │   ├── download_quantized.py  # 下载预量化模型
│   │   ├── efficient_quantize.py  # 高效内存管理的模型量化
│   │   └── model_benchmark.py     # 模型性能对比测试
│   └── vllm-examples/        # vLLM高性能推理示例
│       ├── examples/         # 使用示例
│       ├── vllm/             # vLLM核心实现
│       ├── Dockerfile        # Docker构建文件
│       └── setup.py          # 安装脚本
└── ...
```

## 主要功能模块

### 1. 量化模块 (quantize-examples)

提供在资源受限环境下有效运行大模型的解决方案:

- **下载预量化模型**: 获取已量化的模型以节省本地计算资源
- **高效量化**: 针对Windows环境优化的量化方案
- **性能对比**: 对比原始模型与量化模型的性能差异

### 2. vLLM高性能推理 (vllm-examples)

基于vLLM的高性能LLM推理引擎实现:

- **PagedAttention优化**: 使用高效内存管理技术
- **API服务**: 提供REST API接口服务模型
- **Docker支持**: 容器化部署模型服务
- **本地模型支持**: 支持本地模型和HuggingFace模型

## 使用指南

### 量化模型

```bash
# 下载预量化模型
python deepSeek-learn/quantize-examples/download_quantized.py

# 高效量化自定义模型
python deepSeek-learn/quantize-examples/efficient_quantize.py --model_path <模型路径> --quantize_type dynamic_int8

# 性能对比测试
python deepSeek-learn/quantize-examples/model_benchmark.py --original <原始模型路径> --quantized <量化后模型路径>
```

### vLLM推理

```bash
# 运行API服务器
python deepSeek-learn/vllm-examples/examples/run_model.py --api-mode

# 使用客户端
python deepSeek-learn/vllm-examples/examples/client.py --prompt "请简要介绍量子计算的原理"
```

### Docker运行

```bash
# 构建vLLM镜像
cd deepSeek-learn/vllm-examples
docker build -t vllm-demo .

# 运行容器
docker run --gpus all -p 8000:8000 -v /path/to/models:/app/models vllm-demo
```

## 环境需求

- Python 3.8+
- PyTorch 2.0+
- CUDA支持(推荐用于高性能推理)
- 其他依赖见各目录下的requirements.txt

## 贡献指南

欢迎提交问题报告或功能请求。如需贡献代码，请先fork本仓库，然后提交pull request。

