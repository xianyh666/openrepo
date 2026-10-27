"""LangChain 通过 OpenAI 兼容接口调用火山方舟 Coding Plan。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从项目根目录的 .env 文件载入 ARK_API_KEY。
project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

api_key = os.getenv("ARK_API_KEY")
if not api_key:
    raise RuntimeError("未找到 ARK_API_KEY，请在项目根目录的 .env 文件中填写火山方舟 Coding Plan API Key。")

base_url = os.getenv("ARK_BASE_URL")
if not base_url:
    raise RuntimeError("未找到 ARK_BASE_URL，请在项目根目录的 .env 文件中配置 Coding Plan Base URL。")

model = ChatOpenAI(
    model="ark-code-latest",
    api_key=api_key,
    base_url=base_url,
    # 明确使用 Coding Plan 支持的 Chat Completions 接口
    use_responses_api=False,
)

# # invoke 返回的是 AIMessage，打印 content 即模型生成的文本
# res = model.invoke(input="你是谁呀，能做什么？")
# print(res.content)

# stream 逐块返回，chunk.content 是当前这一小段文本
for chunk in model.stream(input="你是谁呀，能做什么？"):
    print(chunk.content, end="", flush=True)
