# LLM-LearnNote

大型语言模型(LLM)学习与使用工具集，一个帮助用户设置环境、下载模型并进行本地部署的工具包。

## 项目简介

本项目致力于简化本地LLM模型的配置、下载和使用过程，特别针对Windows环境下的CUDA配置和开源LLM模型的使用。它提供了一系列工具脚本，帮助用户解决环境配置、GPU检测、模型下载和验证等常见问题。

## 项目结构

```
llm-LearnNote/
│
├── config/                   # 全局配置目录
│   ├── __init__.py          # 使config成为Python包
│   └── config.py            # 全局配置文件，存储所有配置参数
│
├── util/                     # 工具脚本目录
│   ├── fix_pytorch_cuda12.py # CUDA环境配置工具
│   ├── gpu_test.py          # GPU功能测试工具
│   ├── download_model.py    # 模型下载工具
│   └── verify_model.py      # 模型验证工具
│
├── deepSeek-learn/          # DeepSeek模型学习笔记与应用脚本
│
└── venv/                    # Python虚拟环境(自动生成)
```

## 核心组件

### 配置模块 (config/)

该模块集中管理项目中的所有配置参数，使配置更加统一和可维护。

- **config.py**: 包含所有配置项，如模型路径、CUDA路径、下载源等，每个配置都有详细的中文注释说明用途。

### 工具脚本 (util/)

#### CUDA环境配置 (fix_pytorch_cuda12.py)

用于配置PyTorch与CUDA 12.1的兼容环境，包括：

- 卸载现有PyTorch版本
- 设置CUDA环境变量
- 安装支持CUDA 12.1的PyTorch版本
- 验证PyTorch与CUDA的正确集成

#### GPU测试工具 (gpu_test.py)

测试GPU功能是否正常工作，包括：

- 检测CUDA可用性
- 显示GPU信息（型号、显存等）
- 进行GPU矩阵乘法性能测试
- 对比CPU与GPU计算性能差异

#### 模型下载工具 (download_model.py)

从Hugging Face下载预训练LLM模型，功能包括：

- 使用镜像源加速下载
- 自动创建存储目录
- 下载模型文件
- 加载分词器和模型进行验证

#### 模型验证工具 (verify_model.py)

验证下载的模型是否可以正确加载和使用，包括：

- 验证必要文件是否存在
- 加载分词器
- 检查GPU显存并选择合适的加载方式
- 加载模型
- 进行简单推理测试

### DeepSeek学习模块 (deepSeek-learn/)

用于存放DeepSeek相关的学习笔记和应用脚本，帮助用户学习和使用DeepSeek模型。

## 使用方法

### 环境配置

1. 首先配置CUDA环境：

```bash
python util/fix_pytorch_cuda12.py
```

2. 测试GPU是否可用：

```bash
python util/gpu_test.py
```

### 下载和使用模型

1. 下载预训练模型：

```bash
python util/download_model.py
```

2. 验证模型是否正确下载并可用：

```bash
python util/verify_model.py [可选：模型路径]
```

## 配置说明

所有配置参数都集中在`config/config.py`文件中，包括：

- Hugging Face相关配置（镜像站点、离线模式等）
- CUDA相关配置（安装路径、PyTorch版本等）
- 模型配置（默认模型名称、加载参数等）
- 路径配置（下载目录等）

您可以根据自己的需要修改这些配置。

## 注意事项

- 本项目针对Windows环境设计，特别是CUDA配置部分
- 使用前请确保有足够的磁盘空间和显存
- 大型模型可能需要较长时间下载，请保持网络连接稳定
- 建议使用虚拟环境运行本项目，避免依赖冲突

## 技术依赖

- Python 3.10.9
- PyTorch 2.1.2+cu121
- transformers 4.38.0
- huggingface_hub
- CUDA 12.1

# DeepSeek-Learn 项目

这是一个用于学习和使用DeepSeek系列大语言模型的工具包。本项目提供了简单易用的下载、验证和对话接口，帮助用户快速上手DeepSeek模型。

## 项目结构

```
.
├── deepSeek-learn/       # DeepSeek学习模块
│   ├── examples/         # 示例代码
│   └── README.md         # 模块说明
```

## 功能特点

- 模型下载：支持从Hugging Face下载DeepSeek系列模型
- 模型验证：验证模型是否正确下载和可用
- 交互对话：提供简单的命令行对话接口
- 流式输出：实时显示模型生成内容
- 多轮对话：支持上下文管理，实现多轮对话

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

```bash
python util/download_model.py
```

### 3. 验证模型

```bash
python util/verify_model.py
```

### 4. 开始对话

```bash
python deepSeek-learn/examples/simple_chat.py
```

## 配置说明

您可以通过修改`config/config.py`文件来自定义：

- 使用的模型（默认为DeepSeek-R1-Distill-Qwen-7B）
- 模型下载路径
- 模型加载参数

## 贡献指南

欢迎贡献代码或提出问题！请通过GitHub Issue或Pull Request参与项目改进。
