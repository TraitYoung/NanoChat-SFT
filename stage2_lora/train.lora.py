import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# 1. 配置路径
model_id = "./qwen/Qwen2-0___5B-Instruct"
data_path = "qwen_sft.json"
output_dir = "./qwen2_lora_result"

# 2. 【核心修复】先定义 Tokenizer，确保后面 Collator 能认出来
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right' # 工业级微调标准对齐方式

# 3. 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 4. LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 数据处理函数（保持原有的 Mask 逻辑）
def process_func(example):
    MAX_LENGTH = 512
    # 构造 ChatML 模版
    instruction = tokenizer(f"<|im_start|>system\n你是由陛下亲自训练的私人秘书Bina。<|im_end|>\n<|im_start|>user\n{example['messages'][0]['content']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(f"{example['messages'][1]['content']}<|im_end|>", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    # Prompt 部分设为 -100 (不计损失)
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 加载数据
dataset = load_dataset("json", data_files=data_path, split="train")
tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

# 6. 【关键修复】使用 DataCollatorForSeq2Seq 解决 Batch 长度不一报错
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

# 7. 训练参数设置
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    bf16=True,
    remove_unused_columns=False # 确保手动构造的 labels 不被误删
)

# 8. 启动 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(output_dir)
print("微调完成！权重已安全降落在 qwen2_lora_result 文件夹中。")