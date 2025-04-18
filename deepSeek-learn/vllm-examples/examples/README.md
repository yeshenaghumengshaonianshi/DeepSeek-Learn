# vLLM 使用示例

本目录包含使用 vLLM 引擎的示例脚本，展示如何高效地运行和部署大型语言模型。

## 可用示例

1. **run_model.py** - 基本的模型运行示例，支持多种配置选项
   - 同步推理
   - API服务器模式
   - 性能测量

## 示例用法

### 运行基本推理

```bash
python run_model.py --model deepseek-ai/deepseek-coder-7b-instruct --temperature 0.7 --max-tokens 1024
```

### 运行API服务器

```bash
python run_model.py --model deepseek-ai/deepseek-coder-7b-instruct --api-mode --host 0.0.0.0 --port 8000
```

### 使用多GPU进行张量并行

```bash
python run_model.py --model deepseek-ai/deepseek-llama-67b-chat --tensor-parallel-size 4
```

## 脚本参数

`run_model.py` 支持的命令行参数：

- `--model`: 要加载的模型名称或路径
- `--tensor-parallel-size`: 用于张量并行的GPU数量
- `--gpu-memory-utilization`: 目标GPU内存利用率
- `--max-tokens`: 生成的最大token数
- `--temperature`: 采样温度
- `--prompts`: 要处理的提示或JSON文件路径
- `--api-mode`: 启动API服务器模式

## 提示文件格式

使用JSON文件提供多个提示：

```json
[
  "请介绍一下vLLM的特点和优势。",
  "如何优化LLM的推理性能？",
  "什么是PagedAttention技术？"
]
```

## 后续步骤

1. 尝试使用不同的模型和配置
2. 调整参数以获得最佳性能
3. 参考API服务器代码创建自定义接口 