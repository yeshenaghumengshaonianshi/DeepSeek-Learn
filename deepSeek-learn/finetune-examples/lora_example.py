"""
DeepSeek模型LoRA微调示例

本示例展示如何使用LoRA技术高效微调DeepSeek模型，使其适应特定任务。
使用方法: python deepSeek-learn/finetune-examples/lora_example.py
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(deepseek_dir)
sys.path.insert(0, project_root)

# 导入配置
from config.config import (
    get_default_download_dir,
    DEFAULT_MODEL_NAME
)

# 微调参数
MAX_LENGTH = 512           # 序列最大长度
LORA_RANK = 8              # LoRA秩
LORA_ALPHA = 16            # LoRA缩放参数
LORA_DROPOUT = 0.05        # LoRA随机丢弃率
BATCH_SIZE = 4             # 训练批次大小
GRADIENT_ACCUMULATION = 4  # 梯度累积步数
LEARNING_RATE = 2e-4       # 学习率
NUM_EPOCHS = 3             # 训练轮数
OUTPUT_DIR = os.path.join(deepseek_dir, "models", "lora_finetune")  # 输出目录

# 示例提示词模板
PROMPT_TEMPLATE = """
### 指令:
{instruction}

### 回答:
"""

def prepare_sample_data():
    """准备示例训练数据
    
    在实际应用中，您应该准备自己的领域数据集。
    这里仅使用一个小样本集合作为示例。
    """
    # 创建一个简单的指令数据集
    sample_data = [
        {"instruction": "解释什么是深度学习", "response": "深度学习是机器学习的一个分支，它使用多层神经网络模拟人脑学习过程..."},
        {"instruction": "写一个快速排序算法", "response": "以下是Python实现的快速排序算法:\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n```"},
        {"instruction": "总结量子力学的核心概念", "response": "量子力学的核心概念包括:\n1. 波粒二象性\n2. 测不准原理\n3. 概率解释\n4. 叠加态\n5. 量子纠缠\n这些概念共同构成了理解微观世界的基础框架。"},
        {"instruction": "解释区块链技术的工作原理", "response": "区块链是一种分布式账本技术，通过一系列加密和共识机制确保数据安全..."},
        {"instruction": "如何有效进行时间管理", "response": "有效的时间管理包括：\n1. 设定明确的优先级\n2. 使用番茄工作法\n3. 减少会议时间\n4. 批处理类似任务\n5. 学会委派和拒绝"}
    ]
    
    # 处理数据：添加提示词模板
    processed_data = []
    for item in sample_data:
        prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"])
        processed_data.append({
            "text": prompt + item["response"]
        })
    
    # 转换为HuggingFace数据集格式
    from datasets import Dataset
    return Dataset.from_list(processed_data)

def preprocess_function(examples, tokenizer):
    """对数据进行预处理和分词"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

def main():
    # 获取模型路径
    model_path = get_default_download_dir(DEFAULT_MODEL_NAME)
    print(f"使用模型: {model_path}")
    
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载基础模型（FP16）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 3. 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,                    # LoRA秩
        lora_alpha=LORA_ALPHA,          # LoRA缩放
        lora_dropout=LORA_DROPOUT,      # 丢弃率
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 靶向模型中的模块
        bias="none",
    )
    
    # 4. 准备模型用于训练
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 显示可训练参数比例
    
    # 5. 准备训练数据
    print("准备训练数据...")
    train_dataset = prepare_sample_data()
    
    # 6. 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=True,
        report_to="none",
    )
    
    # 7. 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # 不使用掩码语言建模
    )
    
    # 8. 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 9. 训练模型
    print("开始训练模型...")
    trainer.train()
    
    # 10. 保存微调后的模型
    print(f"保存模型到 {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 11. 测试微调后的模型
    print("\n测试微调后的模型...")
    test_prompt = PROMPT_TEMPLATE.format(instruction="解释强化学习的基本原理")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成结果: \n{generated_text}")
    
    print("\nLoRA微调完成!")

if __name__ == "__main__":
    main() 