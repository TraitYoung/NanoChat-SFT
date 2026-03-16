import json
from bs4 import BeautifulSoup
import html

def parse_html_to_roles(html_content):
    """从HTML中提取所有 [角色名]: 的发言，返回拼接后的文本"""
    # 解码HTML实体（如 &lt; 等）
    text = html.unescape(html_content)
    # 用BeautifulSoup提取纯文本，保留换行
    soup = BeautifulSoup(text, 'html.parser')
    plain_text = soup.get_text(separator='\n')
    lines = plain_text.split('\n')
    roles_lines = []
    for line in lines:
        line = line.strip()
        # 匹配以 [ 开头、包含 ]: 的行（例如 [Chizheng]: 结论先行...）
        if line.startswith('[') and ']:' in line:
            roles_lines.append(line)
    if not roles_lines:
        # 如果没有找到任何角色行，返回整个纯文本（但通常不会）
        return plain_text.strip()
    return '\n'.join(roles_lines)

def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)   # 假设JSON文件是一个列表

    pairs = []
    for item in data:
        # 提取用户消息（title字段）
        title = item.get('title', '').strip()
        if not title:
            continue
        # 提取助手回复（safeHtmlItem中的html）
        html_items = item.get('safeHtmlItem', [])
        if not html_items:
            continue
        html_content = html_items[0].get('html', '')
        if not html_content:
            continue
        response = parse_html_to_roles(html_content)
        if response:
            pairs.append({
                "prompt": title,
                "response": response
            })

    # 写入 JSON Lines 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"提取了 {len(pairs)} 条对话对，已保存到 {output_path}")

if __name__ == '__main__':
    process_json_file('chat.json', 'sft_pairs.jsonl')
