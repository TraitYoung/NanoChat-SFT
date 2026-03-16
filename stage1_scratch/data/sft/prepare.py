import os
import pickle
import numpy as np
import json
from tqdm import tqdm
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# 读取清洗后的对话对
pairs = []
with open('sft_pairs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        pairs.append(json.loads(line))

split_idx = int(len(pairs) * 0.9)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

def process_pairs(pair_list):
    ids = []
    mask = []
    for p in tqdm(pair_list):
        prompt = p['prompt']
        response = p['response']
        full_text = prompt + '\n' + response + '<|endoftext|>'
        tokens = enc.encode(full_text, allowed_special={"<|endoftext|>"})
        prompt_len = len(enc.encode(prompt + '\n'))
        mask.extend([0] * prompt_len + [1] * (len(tokens) - prompt_len))
        ids.extend(tokens)
    return np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)

print("处理训练集...")
train_ids, train_mask = process_pairs(train_pairs)
print("处理验证集...")
val_ids, val_mask = process_pairs(val_pairs)

train_ids.tofile('data/sft/train.bin')
val_ids.tofile('data/sft/val.bin')
np.save('data/sft/train_mask.npy', train_mask)
np.save('data/sft/val_mask.npy', val_mask)

meta = {'vocab_size': enc.n_vocab}
with open('data/sft/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print(f"训练集 token 数: {len(train_ids)}, 验证集 token 数: {len(val_ids)}")
