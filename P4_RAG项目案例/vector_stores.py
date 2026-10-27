from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})


if __name__ == '__main__':
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from langchain_openai import OpenAIEmbeddings

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
        model=config.embedding_model_name,
        api_key=api_key,
        base_url=base_url,
        # 非 OpenAI 服务商不支持 token 格式输入，关闭长度检查，直接发送原始文本
        check_embedding_ctx_length=False,
    )

    retriever = VectorStoreService(embedding).get_retriever()

    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)

