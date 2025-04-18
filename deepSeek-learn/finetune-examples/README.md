# DeepSeek模型微调示例

本目录包含DeepSeek大语言模型的不同微调方法示例，帮助您根据特定任务或领域调整预训练模型，提高模型在特定场景下的表现。

## 什么是模型微调？

模型微调是在预训练模型的基础上，使用特定领域或任务的数据进一步训练模型，使其适应特定应用场景的过程。

## 主要微调方法

### 1. 全参数微调 (Full Fine-tuning)

更新模型的所有参数：

- **优点**：通常能获得最佳性能
- **缺点**：需要大量计算资源，容易过拟合
- **适用场景**：有足够计算资源，且有大量领域数据

### 2. 参数高效微调 (PEFT - Parameter-Efficient Fine-Tuning)

只更新部分参数或引入少量新参数：

#### LoRA (Low-Rank Adaptation)

- 在原始权重矩阵旁添加小的低秩矩阵
- 只训练这些低秩矩阵，冻结原始权重
- 大幅减少训练参数量和内存需求
- 适合大多数微调场景

#### QLoRA (Quantized LoRA)

- 将基础模型量化为较低精度（如4位）
- 在量化模型上应用LoRA
- 进一步降低内存需求，适合消费级硬件

#### Prefix Tuning & P-Tuning

- 在输入序列前添加可训练的前缀向量
- 只训练这些前缀向量
- 参数量极少，适合特定任务调优

#### Adapter Methods

- 在模型层之间插入小型可训练模块
- 其余参数冻结不变
- 模块化设计，易于切换任务

### 3. 提示学习 (Prompt Learning)

通过设计更好的提示模板来优化模型输出：

- **硬提示 (Hard Prompts)**：手动设计固定文本提示
- **软提示 (Soft Prompts)**：在连续空间学习最佳提示嵌入

## 微调方法选择建议

| 方法 | 推荐场景 | 参数量 | 计算需求 | 性能 |
|------|----------|------|------|------|
| 全参数微调 | 资源充足，追求最高性能 | 全部 | 极高 | 最佳 |
| LoRA | 资源有限，通用场景 | <1% | 中等 | 接近全参数 |
| QLoRA | 消费级硬件 | <1% | 低 | 良好 |
| Prefix Tuning | 多任务切换 | <0.1% | 极低 | 中等 |
| 提示学习 | 简单任务适应 | 0 | 最低 | 受限 |

## 本目录示例

1. `lora_example.py` - 使用LoRA微调DeepSeek模型
2. `qlora_example.py` - 使用QLoRA在低端硬件上微调DeepSeek
3. `instruction_tuning.py` - 指令微调示例
4. `multi_turn_conversation.py` - 多轮对话微调示例

## 参考资源

- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [DeepSeek微调指南](https://github.com/deepseek-ai/DeepSeek-LLM) 