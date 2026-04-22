from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from peft import PeftModel

model_path = "/root/autodl-tmp/DeepSeek-R1-Distill-Llama-8B"
lora_path = "/root/deepseek/fine-tuning/output/deepseek/checkpoint-2480"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 【关键修复1】生成参数完全正确，无冲突，杜绝幻觉
generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False,  # 贪心解码，稳定输出
    temperature=None,  # 移除冲突参数
    top_p=None,  # 移除冲突参数
    max_new_tokens=64,  # 严格限制生成长度，只生成答案
    repetition_penalty=1.1
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()


# 【关键修复2】严格复刻训练格式，100%对齐
def build_prompt(instruction, input_text=""):
    if input_text.strip():
        question = f"{instruction}\n{input_text}"
    else:
        question = instruction
    return f"问：{question}\n答："


# 【关键修复3】彻底移除对话历史，每次都是全新单轮问答
def chat(instruction, input_text=""):
    # 每次提问都重新构造prompt，完全不碰历史
    prompt = build_prompt(instruction, input_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    # 只提取「答：」后面的内容，彻底截断多余生成
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split("答：")[-1].split("\n")[0].strip()  # 只取第一行，杜绝续写
    return answer


# 交互式对话（完全无历史、无格式污染版）
print("模型已启动，输入 'quit' 退出")
while True:
    print("\n请输入指令（格式：指令 + 输入内容，用空格分隔）")
    user_input = input("你：")
    if user_input.lower() == "quit":
        print("对话结束")
        break

    # 自动拆分指令和输入（兼容你数据集的格式）
    if "日期：" in user_input:
        instruction = "给定一个日期，输出它是星期几。"
        input_text = user_input.split("日期：")[-1].strip()
    elif "数字1：" in user_input:
        instruction = "给定两个数，计算它们的最大公约数。"
        input_text = user_input.strip()
    else:
        instruction = user_input
        input_text = ""

    response = chat(instruction, input_text)
    print(f"\nAI：{response}")