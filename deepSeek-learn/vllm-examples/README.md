# vLLM 示例

这个目录包含使用 vLLM 进行高效大语言模型推理的示例代码和工具。

## 什么是 vLLM?

vLLM 是一个高性能、高吞吐量的 LLM 推理和服务引擎，具有以下特点：

- **PagedAttention**：基于 KV 缓存的分页内存管理，提高内存效率
- **连续批处理**：动态调度请求并行处理，提高吞吐量
- **量化支持**：支持 INT8/INT4 量化，降低内存和计算需求
- **并行解码**：高效张量并行和推理优化
- **流式输出**：支持流式生成和输出

## 目录结构

```
vllm-examples/
├── Dockerfile          # Docker 构建文件
├── setup.py            # 安装脚本
├── examples/           # 示例脚本
│   ├── run_model.py    # 模型运行示例
│   └── client.py       # API客户端示例
└── vllm/               # vLLM 实现
    ├── __init__.py     # 包初始化
    ├── config.py       # 配置文件
    ├── engine/         # 推理引擎实现
    ├── entrypoints/    # 命令行和API入口
    ├── outputs.py      # 输出数据类
    ├── sampling_params.py # 采样参数
    └── utils.py        # 工具函数
```

## 配置系统

本项目使用集中配置管理，主要配置位于 `vllm/config.py` 文件中：

- **模型配置**：默认模型路径和可用模型列表
- **API配置**：服务器主机、端口和其他API参数
- **推理配置**：推理相关的默认参数设置

### 默认模型设置

当前默认使用本地模型：`models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

可以通过以下方式指定其他模型：
1. 使用模型简称：`--model deepseek-coder`
2. 使用本地路径：`--model /path/to/model`
3. 使用HuggingFace模型ID：`--model deepseek-ai/deepseek-llm-7b-chat`

## 快速开始

### 1. 安装依赖

安装 vLLM 和依赖项：

```bash
pip install -e .
```

### 2. 运行模型

使用示例脚本运行模型：

```bash
# 使用默认模型
python examples/run_model.py

# 使用指定模型
python examples/run_model.py --model deepseek-coder
```

更多参数选项：

```bash
python examples/run_model.py --help
```

### 3. 启动API服务器

启动 vLLM API 服务器：

```bash
python examples/run_model.py --api-mode
```

或者指定特定模型：

```bash
python examples/run_model.py --model deepseek-coder --api-mode --host 0.0.0.0 --port 8000
```

### 4. 使用客户端

使用客户端测试API：

```bash
# 基本使用
python examples/client.py

# 查看可用模型
python examples/client.py --list-models

# 使用流式输出
python examples/client.py --stream --prompt "请用中文解释量子计算原理"
```

## 使用 Docker

### 1. 构建镜像

```bash
docker build -t vllm-demo .
```

### 2. 运行容器

```bash
# 挂载本地模型目录
docker run --gpus all -p 8000:8000 -v /path/to/models:/app/models vllm-demo
```

使用自定义模型：

```bash
docker run --gpus all -p 8000:8000 -v /path/to/models:/app/models vllm-demo python examples/run_model.py --model deepseek-coder --api-mode
```

## API 使用示例

生成文本 (非流式)：

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请介绍一下vLLM的主要特点",
    "sampling_params": {
      "temperature": 0.7,
      "max_tokens": 100
    }
  }'
```

流式生成：

```bash
curl -X POST http://localhost:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请介绍一下vLLM的主要特点",
    "sampling_params": {
      "temperature": 0.7,
      "max_tokens": 100
    }
  }'
```

## 高级配置

### 张量并行

在多GPU环境中使用张量并行提高性能：

```bash
python examples/run_model.py --tensor-parallel-size 2
```

### 内存优化

调整GPU内存利用率：

```bash
python examples/run_model.py --gpu-memory-utilization 0.75
```

### 自定义配置文件

可以直接修改 `vllm/config.py` 文件中的配置选项，定制您自己的默认设置。

## 性能优化提示

1. 使用较新的GPU获取最佳性能
2. 对于大型模型使用张量并行
3. 适当调整GPU内存利用率
4. 使用批处理模式处理多个请求
5. 考虑使用量化减少内存需求

## 参考文档

- [vLLM 官方文档](https://github.com/vllm-project/vllm)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [用于缓存优化的张量并行技术](https://arxiv.org/abs/2310.12082) 