import json

# 读取你原本的 18MB 语料
src_file = 'sft_pairs.jsonl' 
dst_file = 'qwen_sft.json'

qwen_data = []
with open(src_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        # 转换为 Qwen 要求的对话格式
        qwen_data.append({
            "messages": [
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['response']}
            ]
        })

with open(dst_file, 'w', encoding='utf-8') as f:
    json.dump(qwen_data, f, ensure_ascii=False, indent=2)

print(f"转换完成！已生成 {dst_file}")