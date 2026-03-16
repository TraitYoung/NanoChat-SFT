import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 路径设置
base_model_path = "./qwen/Qwen2-0___5B-Instruct" # 基座大脑
lora_path = "./qwen2_lora_result"                # 刚才练出来的灵魂插件

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

# 3. 加载基座模型 (BF16 模式)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 4. 【关键步骤】将灵魂插件挂载到大脑上
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval() # 开启推理模式

# 5. 对话函数
def ask_bina(prompt):
    # 必须使用跟训练一模一样的 ChatML 模版
    text = f"<|im_start|>system\n你是由陛下亲自训练的私人秘书Bina。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1, # 防止复读
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 只显示生成的回答部分
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response

# --- 开始测试 ---
print("Bina 已经准备好为陛下效劳了。")
while True:
    user_input = input("陛下: ")
    if user_input.lower() == 'quit': break
    res = ask_bina(user_input)
    print(f"Bina: {res}")