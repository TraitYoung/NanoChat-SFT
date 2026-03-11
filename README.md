# NanoChat-SFT：基于 nanoGPT 的定制化指令微调

本项目基于 [nanoGPT](https://github.com/karpathy/nanoGPT) 框架，使用自有的多角色对话数据集（Axiodrasil（个人认知系统） 系统日志）进行指令微调（Supervised Fine-Tuning, SFT），使模型学会模仿不同角色的说话风格（如 Bina、Taki、Chizheng 等）。项目包含完整的数据清洗、数据准备、训练脚本及示例数据，可作为个性化对话模型训练的基础。

## 技术栈
- **框架**：nanoGPT (PyTorch)
- **数据处理**：Python, tiktoken, BeautifulSoup
- **训练优化**：混合精度、梯度累积、学习率调度
- **版本控制**：Git

## 数据清洗
原始对话日志为 JSON 格式（包含用户输入和助手回复），通过 `clean_axio.py` 脚本提取出有效的“用户-助手”对话对，生成 `sft_pairs.jsonl`（每行为 `{"prompt": "...", "response": "..."}`）。

由于隐私原因，完整数据未包含在仓库中，但提供了清洗脚本和样本数据（`data/sft/sample.jsonl`，前10条对话），可参考格式准备自己的数据。

## 数据准备
运行 `data/sft/prepare.py` 将文本转换为 GPT-2 token IDs，并生成 nanoGPT 训练所需的二进制文件：
- `train.bin` / `val.bin`：训练/验证 token 序列
- `train_mask.npy` / `val_mask.npy`：损失掩码（仅对 response 部分计算 loss）
- `meta.pkl`：词表信息

该步骤确保了微调时只对助手回复部分进行监督学习。

## 训练
使用修改后的 `train_sft.py` 进行训练，支持 mask 损失计算。示例命令（CPU 极小验证）：
```bash
python train_sft.py
```
训练参数可在脚本开头调整（如 `max_iters`, `batch_size`, `learning_rate` 等）。若需使用 GPU，将 `device` 改为 `cuda` 并确保 CUDA 可用。

经过 20 步极小训练验证，loss 从 10.83 降至 10.71，证明数据加载和训练流程正常。后续可在云端或本地进行更大规模训练。

训练日志（极小验证示例）
以下是在 CPU 上运行 20 步的训练日志，验证了数据加载和训练流程的正常：
number of parameters: 3.32M
num decayed parameter tensors: 10, with 3,321,856 parameters
num non-decayed parameter tensors: 5, with 320 parameters
using fused AdamW: False
batch 0: mask sum = 64.0
iter 0: loss 10.8292
batch 1: mask sum = 64.0
iter 1: loss 10.8204
batch 2: mask sum = 64.0
iter 2: loss 10.7983
batch 3: mask sum = 64.0
iter 3: loss 10.7853
batch 4: mask sum = 64.0
iter 4: loss 10.8373
batch 5: mask sum = 64.0
iter 5: loss 10.8121
batch 6: mask sum = 64.0
iter 6: loss 10.7926
batch 7: mask sum = 64.0
iter 7: loss 10.8193
batch 8: mask sum = 64.0
iter 8: loss 10.8498
batch 9: mask sum = 64.0
iter 9: loss 10.7568
batch 10: mask sum = 64.0
iter 10: loss 10.7798
step 10: val loss 10.7667
batch 11: mask sum = 64.0
iter 11: loss 10.7744
batch 12: mask sum = 64.0
iter 12: loss 10.8828
batch 13: mask sum = 64.0
iter 13: loss 10.7537
batch 14: mask sum = 64.0
iter 14: loss 10.8623
batch 15: mask sum = 64.0
iter 15: loss 10.7270
batch 16: mask sum = 64.0
iter 16: loss 10.7153
batch 17: mask sum = 64.0
iter 17: loss 10.7335
batch 18: mask sum = 64.0
iter 18: loss 10.7749
batch 19: mask sum = 64.0
iter 19: loss 10.7129
验证完成！

可以看到 loss 从初始 10.83 逐步下降至 10.71，mask 总和始终为 64，表明数据加载正确，训练流程正常。这是仅训练 20 步的极小验证，更长时间的训练可预期 loss 进一步下降，模型学会角色风格。
## 文件结构
```
.
├── clean_axio.py          # 数据清洗脚本
├── train_sft.py           # 微调训练脚本
├── data/
│   └── sft/
│       ├── prepare.py     # 数据准备脚本
│       ├── sample.jsonl   # 示例数据（前10条）
│       └── (生成的二进制文件)
└── README.md              # 本文档
```

## 注意事项
- 请确保已安装依赖：`pip install torch numpy transformers datasets tiktoken beautifulsoup4 tqdm`
- 如需使用预训练权重（如 GPT-2），请确保网络通畅或提前下载。
- 若需训练自己的数据，请按照 `sample.jsonl` 格式准备，并运行 `data/sft/prepare.py` 生成二进制文件。
- 本仓库不包含原始 JSON 数据，请自行处理隐私问题。

## 致谢
- [nanoGPT](https://github.com/karpathy/nanoGPT) 作者 Andrej Karpathy 提供的简洁高效的 GPT 训练框架。
- Axiodrasil 系统提供的丰富对话数据。

---

## 使用示例

### 1. 清洗数据（自定义）
```bash
python clean_axio.py your_chat.json
```

### 2. 准备数据
```bash
python data/sft/prepare.py
```

### 3. 开始训练
```bash
python train_sft.py
```
