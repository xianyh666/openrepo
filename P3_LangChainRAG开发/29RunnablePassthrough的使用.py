"""
提示词：用户的提问 + 向量库中检索到的参考资料
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 从项目根目录的 .env 文件载入 ARK_API_KEY。
project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

api_key = os.getenv("ARK_API_KEY")
if not api_key:
    raise RuntimeError("未找到 ARK_API_KEY，请在项目根目录的 .env 文件中填写火山方舟 Coding Plan API Key。")

base_url = os.getenv("ARK_BASE_URL")
if not base_url:
    raise RuntimeError("未找到 ARK_BASE_URL，请在项目根目录的 .env 文件中配置 Coding Plan Base URL。")

# 创建嵌入模型对象
# model 需要填你在火山方舟控制台开通的嵌入模型名（如 doubao-embedding-large）或推理接入点 ID（如 ep-xxxx）
embedding = OpenAIEmbeddings(
    model="doubao-embedding-vision",
    api_key=api_key,
    base_url=base_url,
    # 非 OpenAI 服务商不支持 token 格式输入，关闭长度检查，直接发送原始文本
    check_embedding_ctx_length=False,
)


def print_prompt(prompt):
    print(prompt.to_string())
    print("=" * 20)
    return prompt


model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=api_key,
    base_url=base_url,
    # 明确使用 Coding Plan 支持的 Chat Completions 接口
    use_responses_api=False,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
        ("user", "用户提问：{input}")
    ]
)

vector_store = InMemoryVectorStore(embedding=embedding)

# 准备一下资料（向量库的数据）
# add_texts 传入一个 list[str]
vector_store.add_texts(
    ["减肥就是要少吃多练", "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"])

input_text = "怎么减肥？"

# langchain中向量存储对象，有一个方法：as_retriever，可以返回一个Runnable接口的子类实例对象
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


def format_func(docs: list[Document]):
    if not docs:
        return "无相关参考资料"

    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"

    return formatted_str

# chain
chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | print_prompt | model | StrOutputParser()
)

res = chain.invoke(input_text)
print(res)
"""
retriever:
    - 输入：用户的提问       str
    - 输出：向量库的检索结果  list[Document]
prompt:
    - 输入：用户的提问 + 向量库的检索结果   dict
    - 输出：完整的提示词                 PromptValue
"""
