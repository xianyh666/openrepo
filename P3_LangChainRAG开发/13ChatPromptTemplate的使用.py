"""LangChain 通过 OpenAI 兼容接口调用火山方舟 Coding Plan，演示 ChatPromptTemplate 的使用。"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Windows 控制台默认 GBK 编码，这里强制 UTF-8 输出，避免中文乱码
sys.stdout.reconfigure(encoding="utf-8")

# 从项目根目录的 .env 文件载入 ARK_API_KEY。
project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

api_key = os.getenv("ARK_API_KEY")
if not api_key:
    raise RuntimeError("未找到 ARK_API_KEY，请在项目根目录的 .env 文件中填写火山方舟 Coding Plan API Key。")

base_url = os.getenv("ARK_BASE_URL")
if not base_url:
    raise RuntimeError("未找到 ARK_BASE_URL，请在项目根目录的 .env 文件中配置 Coding Plan Base URL。")

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你现在是新闻工作者，可以创作治国理政的纲领。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一个发言"),
    ]
)

history_data = [
    ("human", "你来写一个治国理政的纲领"),
    ("ai", "坚持以人民为中心的发展思想，推动高质量发展"),
    ("human", "好纲领再来一个"),
    ("ai", "坚持全面深化改革，扩大高水平对外开放"),
]

# StringPromptValue    to_string()
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=api_key,
    base_url=base_url,
    # 明确使用 Coding Plan 支持的 Chat Completions 接口
    use_responses_api=False,
)

# langchain 会丢弃 reasoning_content，所以用底层 OpenAI 客户端发起流式请求，
# 从而实时显示思考链路和最终回答。注意：此版本 model.client 直接就是
# chat.completions 对象，直接调用 .create() 即可。
client = model.client
stream = client.create(
    model=model.model_name,
    messages=[{"role": "user", "content": prompt_text}],
    stream=True,
)

print("【思考链路】")
print("-" * 40)
answer_parts = []
has_reasoning = False
for chunk in stream:
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        has_reasoning = True
        print(reasoning, end="", flush=True)
    if delta.content:
        answer_parts.append(delta.content)
if not has_reasoning:
    print("（本次没有输出思考内容）")
print()
print("-" * 40)
print("【最终回答】")
print("".join(answer_parts))

