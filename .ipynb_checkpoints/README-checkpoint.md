# Bina-Project: SFT 演进实验 (Scratch to Industry)

本项目是一个关于大模型垂直领域微调（SFT）的完整对比实验。记录了从零实现底层 Transformer 架构，到应用工业级 LoRA 技术进行语感对齐的全过程。

## 项目结构 (Project Structure)

```text
.
├── stage1_scratch/            # 阶段一：底层原理探索 (NanoGPT)
│   ├── model.py               # 手写 Transformer 核心代码
│   ├── train_sft.py           # 带有 Loss Mask 的训练脚本
│   ├── sample.py              # 推理与权重加载脚本
│   └── out-sft-test/          # (已忽略) NanoGPT 训练权重存点
├── stage2_lora/               # 阶段二：工业级微调 (Qwen + PEFT)
│   ├── train_lora.py          # 基于 Transformers/PEFT 的微调脚本
│   ├── inference_lora.py      # 合并 LoRA 权重的最终对话引擎
│   └── convert_data.py        # 语料格式对齐工具 (JSONL -> ChatML)
├── logs/                      # 实验证据链
│   ├── nanogpt_3m.log         # 乱码阶段记录
│   ├── nanogpt_124m.log       # 汉字恢复记录
│   └── qwen_lora.log# 最终收敛日志
│── images/
│   ├── qwen_train_loss.jpg    # Qwen 模型训练记录
│   ├── solute_64.jpg          # 参数扩大后 mask_sum 出现波动记录
├── .gitignore                 # 环境屏蔽与数据脱敏
└── README.md                  # 本文档

```

## 三阶段演进记录

### 1. NanoGPT 3M：逻辑验证与编码陷阱

* **配置**：`n_layer=2, n_embd=64`，手动实现 CrossEntropy 掩码。
* **现象**：输出全为乱码（Byte Salad）。
* **结论**：GPT-2 字节级分词（BPE）将汉字拆解为多个 Token。3M 模型容量无法拟合这种高维度的序列依赖，导致 UTF-8 解码崩溃。

> [图片坐标 1：mask_sum 波动截图](images/solute_64.jpg)

### 2. NanoGPT 124M：规模效应与中文表征

* **配置**：`n_layer=12, n_embd=768` (GPT-2 Base 同级规模)。
* **突破**：**成功输出中文**。Loss 从 10.8 降至 3.69。
* **局限**：虽能说人话，但受限于 18MB 语料的知识密度，逻辑容易陷入复读（Repetition）。验证了 Scaling Law 在解决编码问题上的有效性，但也明确了小模型缺乏常识地基的局限。

### 3. Qwen2-0.5B + LoRA：语感对齐与工业落地

* **技术栈**：PEFT (LoRA Rank=16), BF16 混合精度。
* **成果**：**逻辑连贯，语气高度对齐**。Loss 稳定收敛至 2.8 附近。
* **技术点**：通过 `DataCollatorForSeq2Seq` 解决 Batch Padding 冲突，并继承了阶段一中的 Loss Mask 策略，确保模型只学习 Response 部分。

> [图片坐标 2：Qwen 训练 Loss 曲线](images/qwen_train_loss.jpg)

### 4. 实验总结

微调（SFT）能够高效解决“皮囊（语气）”问题，但无法彻底根除 0.5B 小模型在多角色语境下的身份混淆（Identity Confusion）。本项目已达成风格迁移目标，更深层的硬性逻辑约束将在后续项目 **Axiodrasil** 中通过 LangGraph 状态机实现。

## 模型权重 (Model Weights)

经过训练的 LoRA 适配器（Adapters）已托管至 Hugging Face：
[🔗 iShowRelx5/Bina-Qwen2-0.5B-LoRA](https://www.google.com/search?q=https://huggingface.co/iShowRelx5/Bina-Qwen2-0.5B-LoRA)

加载代码示例：

```python
from peft import PeftModel
# 仅需加载适配器，无需全量下载 1GB 权重
model = PeftModel.from_pretrained(base_model, "iShowRelx5/Bina-Qwen2-0.5B-LoRA")

```
