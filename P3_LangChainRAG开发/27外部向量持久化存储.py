"""LangChain 通过 OpenAI 兼容接口调用火山方舟嵌入模型，演示外部向量持久化存储。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader

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

# Chroma 向量数据库（轻量级的）
# 确保 langchain-chroma chromadb 这两个库安装了的，没有的话请pip install

vector_store = Chroma(
    collection_name="test",     # 当前向量存储起个名字，类似数据库的表名称
    embedding_function=embedding,       # 嵌入模型
    persist_directory="./chroma_db"     # 指定数据存放的文件夹
)




# 检索 返回类型list[Document]
result = vector_store.similarity_search(
    "Python是不是简单易学呀",
    3,        # 检索的结果要几个
    filter={"source": "黑马程序员"}
)

print(result)
