# =========================
# DeepSeek AI Agent（本地API对接版 · 完美适配你的部署脚本）
# =========================
from openai import OpenAI
from pathlib import Path

# ========= 【无需修改，直接对接你的本地API】 =========
# 你的vLLM API服务地址
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # 本地服务不需要真实key
)
MODEL_NAME = "deepseek"  # 对应你启动的模型名

# ========= 【LLM生成函数（对接本地API）】 =========
def deepseek_generate(prompt, max_new_tokens=512, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"生成错误：{str(e)}"

# ========= RAG模块 =========
def rag_search(query):
    return f"【知识库】关于 {query} 的相关资料已检索完成"

# ========= 工具函数 =========
def tool_rag(query):
    return rag_search(query)

def tool_calculator(expr):
    try:
        return str(eval(expr))
    except:
        return "计算错误"

# ========= 记忆模块 =========
class Memory:
    def __init__(self, max_len=5):
        self.history = []
        self.max_len = max_len

    def add(self, role, content):
        self.history.append(f"{role}: {content}")
        if len(self.history) > self.max_len:
            self.history.pop(0)

    def get_context(self):
        return "\n".join(self.history)

memory = Memory()

# ========= Agent决策模块 =========
def agent_decide(query):
    prompt = f"""
你是一个AI助手，请判断用户问题类型：
1. 需要知识库查询 → RAG
2. 需要数学计算 → CALC
3. 普通对话 → CHAT

用户问题：{query}
只输出一个：RAG / CALC / CHAT
"""
    decision = deepseek_generate(prompt, max_new_tokens=10, temperature=0.1).strip().upper()
    if "RAG" in decision:
        return "RAG"
    elif "CALC" in decision:
        return "CALC"
    return "CHAT"

# ========= Agent核心逻辑 =========
def agent_run(query):
    memory.add("用户", query)
    decision = agent_decide(query)
    print(f"📌 决策：{decision}")

    if decision == "RAG":
        context = tool_rag(query)
        prompt = f"基于以下知识回答：\n{context}\n问题：{query}"
        result = deepseek_generate(prompt)

    elif decision == "CALC":
        result = tool_calculator(query)

    else:
        context = memory.get_context()
        prompt = f"对话历史：\n{context}\n问题：{query}"
        result = deepseek_generate(prompt)

    memory.add("AI", result)
    return result

# ========= 主程序 =========
if __name__ == "__main__":
    print("\n🤖 DeepSeek AI Agent（本地API版）已启动！输入 exit 退出")
    print("⚠️  请先执行：bash /root/deepseek/chuli/推理微调后的-checkpoint.sh 启动模型服务")
    while True:
        query = input("\n你：").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 结束")
            break
        if not query:
            continue
        response = agent_run(query)
        print("🤖 Agent：", response)ni