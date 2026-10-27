"""LangChain 通过 OpenAI 兼容接口调用火山方舟嵌入模型，演示内存向量存储。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
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

vector_store = InMemoryVectorStore(
    embedding=embedding
)


loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
    source_column="source",     # 指定本条数据的来源是哪里（csv的列）
)

documents = loader.load()
# id1 id2 id3 id4 ...
# 向量存储的 新增、删除、检索
vector_store.add_documents(
    documents=documents,        # 被添加的文档，类型：list[Document]
    ids=["id"+str(i) for i in range(1, len(documents)+1)] # 给添加的文档提供id（字符串）  list[str]
)

# 删除  传入[id, id...]
vector_store.delete(["id1", "id3"])

# 检索 返回类型list[Document]
result = vector_store.similarity_search(
    "股票代码？",
    3       # 检索的结果要几个
)

print(result)
