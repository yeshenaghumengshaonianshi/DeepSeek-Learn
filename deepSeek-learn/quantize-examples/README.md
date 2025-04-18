# DeepSeek 模型量化示例

本目录包含多种DeepSeek模型量化的示例和工具，旨在减少模型体积、加快推理速度和降低内存占用。

## 量化方法概述

量化是通过降低模型权重的精度（例如从FP32降低到INT8或INT4）来减少模型大小和计算需求的技术。DeepSeek模型支持多种量化方法：

1. **PyTorch动态量化**：使用PyTorch内置的量化功能
2. **GGML/GGUF量化**：将模型转换为GGML/GGUF格式（llama.cpp使用的格式）
3. **简易CPU量化**：适用于GPU内存受限环境的低成本量化
4. **混合CPU-GPU量化**：部分在CPU、部分在GPU进行的混合量化
5. **高效内存管理量化**：专为Windows环境优化的量化工具

## 脚本说明

### 1. efficient_quantize.py - 高效内存管理量化

专为Windows环境设计，使用分层量化技术减少内存占用，支持多种量化类型。

```bash
python deepSeek-learn/quantize-examples/efficient_quantize.py --model_path <模型路径> --quantize_type <量化类型> --output_dir <输出目录>
```

量化类型选项:
- `dynamic_int8`: PyTorch动态INT8量化（Windows完全兼容）
- `dynamic_fp16`: Float16半精度量化（Windows完全兼容）
- `ggml_q4`: GGML Q4量化（需要安装ctransformers库）
- `ggml_q8`: GGML Q8量化（需要安装ctransformers库）

### 2. model_benchmark.py - 模型性能对比测试

对比原始模型和量化后模型的性能，包括内存使用、推理速度和输出质量。

```bash
python deepSeek-learn/quantize-examples/model_benchmark.py --original <原始模型路径> --quantized <量化后模型路径> --test_prompts <测试集文件>
```

生成美观的HTML报告，直观展示性能差异。

### 3. torch_quantize_example.py - PyTorch原生量化

使用PyTorch内置量化API进行模型量化。

```bash
python deepSeek-learn/quantize-examples/torch_quantize_example.py
```

### 4. ggml_quantize_example.py - GGML格式量化

将模型转换为GGML/GGUF格式并进行量化。

```bash
python deepSeek-learn/quantize-examples/ggml_quantize_example.py
```

### 5. simple_cpu_quantize.py - 简易CPU量化

在CPU上进行模型量化，适用于GPU内存受限的环境。

```bash
python deepSeek-learn/quantize-examples/simple_cpu_quantize.py
```

### 6. hybrid_quantize.py - 混合CPU-GPU量化

部分层在GPU上量化，部分在CPU上量化，灵活处理内存限制。

```bash
python deepSeek-learn/quantize-examples/hybrid_quantize.py
```

### 7. download_quantized.py - 下载预量化模型

从Hugging Face下载已经量化好的模型。

```bash
python deepSeek-learn/quantize-examples/download_quantized.py
```

## 配置

所有量化脚本使用的配置参数位于`config/config.py`中，可以在该文件中调整默认参数。

## 依赖安装

### 基本依赖
```bash
pip install torch transformers tqdm psutil numpy
```

### GGML/GGUF量化依赖
```bash
pip install ctransformers huggingface_hub[cli]
```

## 常见问题解决

1. **内存不足错误**：尝试使用`efficient_quantize.py`或`simple_cpu_quantize.py`
2. **Windows上的兼容性问题**：避免使用依赖bitsandbytes的量化方法
3. **量化后精度下降**：尝试使用更高精度的量化类型（如从INT4升级到INT8）

## 性能对比

不同量化方法在10B模型上的性能对比：

| 量化方法 | 大小减少 | 内存减少 | 速度提升 | 质量保持 |
|---------|---------|---------|---------|----------|
| FP16    | 50%     | 50%     | 1.2-1.5x| 99%      |
| INT8    | 75%     | 70%     | 1.5-2.0x| 95%      |
| INT4    | 87%     | 80%     | 2.0-3.0x| 85-90%   |
| GGML Q4 | 80%     | 80%     | 3.0-4.0x| 80-90%   |

## 什么是模型量化？

模型量化是将模型权重从高精度数据类型（如FP32或FP16）转换为低精度数据类型（如INT8或INT4）的过程，以减小模型大小并加速推理，同时尽可能保持模型性能。

## 主要量化方案

### 1. 静态量化 (Post-Training Quantization, PTQ)

在训练完成后对模型进行量化，无需重新训练：

- **INT8量化**：将FP16/FP32权重量化为INT8，可减小模型体积约75%，适用于资源受限场景
- **INT4量化**：将权重量化为INT4，可减小模型体积约87.5%，但精度损失更大
- **GPTQ量化**：一种针对大型语言模型优化的量化技术，支持INT4/INT8精度
- **AWQ (Activation-aware Weight Quantization)**：考虑激活值的权重量化方法，性能更好
- **GGUF格式**：llama.cpp使用的通用量化格式，支持多种精度级别

### 2. 量化感知训练 (Quantization-Aware Training, QAT)

在训练过程中模拟量化效果，以减轻量化带来的精度损失：

- 在训练期间模拟量化操作
- 使模型适应量化误差
- 比静态量化精度更高，但需要更多训练资源

### 3. 新兴量化方法

- **SmoothQuant**：通过数据平滑技术改善激活值量化
- **BitQuant**：根据权重重要性动态分配比特数
- **FP8量化**：介于FP16和INT8之间的精度，保持更好的数值范围

## 量化方案选择建议

| 方案 | 推荐场景 | 优点 | 缺点 |
|------|----------|------|------|
| INT8 PTQ | 资源受限、要求性能较好 | 体积减小、性能良好 | 轻微精度损失 |
| INT4 PTQ | 极度资源受限 | 体积显著减小 | 明显精度损失 |
| GPTQ | 需要在消费级GPU上运行大模型 | 体积小、速度快 | 需要额外量化步骤 |
| AWQ | 追求更好性能平衡 | 精度损失小 | 实现较复杂 |
| QAT | 追求最高精度 | 精度最好 | 需要训练资源 |

## 本目录示例

1. `ptq_int8_example.py` - INT8静态量化示例
2. `gptq_example.py` - 使用GPTQ进行INT4量化示例
3. `awq_example.py` - 使用AWQ进行量化示例

## 参考资源

- [HuggingFace Transformers量化文档](https://huggingface.co/docs/transformers/quantization)
- [AWQ: Activation-aware Weight Quantization](https://github.com/mit-han-lab/llm-awq)
- [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## 测试数据

`data`目录包含用于测试和基准测试的示例数据：

1. **test_prompts.txt**: 包含多种类型的短提示，用于模型基准测试，涵盖了知识问答、编程、创意写作等场景。用于测试模型的多方面能力和量化后性能的保持程度。

2. **long_form_test.txt**: 包含长文本生成任务，用于测试模型处理长上下文的能力，特别适合验证量化对长文本生成质量的影响。

使用方法示例：

```bash
# 使用自定义测试集进行基准测试
python deepSeek-learn/quantize-examples/model_benchmark.py --original <原始模型路径> --quantized <量化后模型路径> --test_prompts deepSeek-learn/quantize-examples/data/test_prompts.txt

# 使用长文本测试集
python deepSeek-learn/quantize-examples/model_benchmark.py --original <原始模型路径> --quantized <量化后模型路径> --test_prompts deepSeek-learn/quantize-examples/data/long_form_test.txt
``` 