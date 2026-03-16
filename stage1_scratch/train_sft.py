import os
import numpy as np
import torch
from model import GPTConfig, GPT

# 硬编码参数（适配你的小数据测试）
out_dir = 'out-sft-test'
eval_interval = 200
log_interval = 10
eval_iters = 5
eval_only = False
init_from = 'scratch'
dataset = 'sft'
batch_size = 12
block_size = 512
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 1e-4          # 调低学习率
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5                # 降低梯度裁剪阈值
decay_lr = True
warmup_iters = 100
lr_decay_iters = 20
min_lr = 3e-5
device = 'cuda'
dtype = 'bfloat16'
compile = False
gradient_accumulation_steps = 1

# 数据加载函数
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        mask = np.load(os.path.join(data_dir, 'train_mask.npy'), mmap_mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        mask = np.load(os.path.join(data_dir, 'val_mask.npy'), mmap_mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy((mask[i+1:i+1+block_size]).astype(np.float32)) for i in ix])
    return x.to(device), y.to(device), m.to(device)

# 初始化模型（从头开始）
gptconf = GPTConfig(
    block_size=block_size,
    vocab_size=50304,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias
)
model = GPT(gptconf)
model.to(device)

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

# 训练循环
for iter_num in range(max_iters):
    X, Y, M = get_batch('train')
    
    # 调试：打印 mask 总和，确保不为 0
    mask_sum = M.sum().item()
    print(f"batch {iter_num}: mask sum = {mask_sum}")
    if mask_sum == 0:
        print("警告：mask 全零，跳过此 batch")
        continue

    logits, loss = model(X, Y)
    loss = (loss * M).sum() / mask_sum  # 使用 mask_sum 避免重复计算
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: loss {loss.item():.4f}")
    
    if iter_num % eval_interval == 0 and iter_num > 0:
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(eval_iters):
                X, Y, M = get_batch('val')
                mask_sum = M.sum().item()
                if mask_sum == 0:
                    continue
                logits, loss = model(X, Y)
                loss = (loss * M).sum() / mask_sum
                losses.append(loss.item())
        if losses:
            print(f"step {iter_num}: val loss {np.mean(losses):.4f}")
        else:
            print(f"step {iter_num}: val loss skipped (no valid masks)")
        model.train()

print("验证完成！")

import os

# 确保文件夹存在
os.makedirs(out_dir, exist_ok=True)

# 构造要保存的数据包
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': gptconf,
    'iter_num': iter_num,
    'best_val_loss': loss.item(), # 这里存最后的 loss
    'config': gptconf.__dict__,
}

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"正在保存模型权重至 {ckpt_path}...")
torch.save(checkpoint, ckpt_path)
print("保存成功！现在可以去推理了。")