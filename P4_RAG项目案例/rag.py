import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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


def print_prompt(prompt):
    """分区打印本轮提示词，方便区分检索资料、历史记录和当前问题。"""
    messages = prompt.messages
    print("\n" + "=" * 24 + " 本轮发送给模型的提示词 " + "=" * 24)

    if len(messages) < 3:
        # 提示词结构不符合当前 RAG 模板时，保留通用打印方式。
        print(prompt.to_string())
        print("=" * 72)
        return prompt

    # 当前模板的第一个 System 消息包含系统指令和 {context} 检索结果。
    system_content = messages[0].content
    if isinstance(system_content, str) and "参考资料:" in system_content:
        instruction, retrieved_context = system_content.split("参考资料:", 1)
        print("\n【系统指令】")
        print(instruction.strip())
        print("\n【本轮向量检索结果】")
        print(retrieved_context.strip())
    else:
        print("\n【系统指令与本轮检索资料】")
        print(system_content)

    # 第 2 个消息是提示词模板里的历史说明文字；实际历史在其后。
    history_messages = messages[2:-1]
    print("\n【历史对话｜之前轮次】")
    if history_messages:
        role_names = {
            "human": "用户",
            "ai": "助手",
            "system": "系统",
            "tool": "工具",
        }
        for message in history_messages:
            role = role_names.get(message.type, message.type)
            print(f"{role}：{message.content}")
    else:
        print("（无历史记录，这是本会话的第一轮）")

    current_message = messages[-1]
    print("\n【本轮新问题】")
    print(current_message.content)
    print("\n" + "=" * 72)

    return prompt


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=embedding
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            api_key=api_key,
            base_url=base_url,
            # 明确使用 Coding Plan 支持的 Chat Completions 接口
            use_responses_api=False,
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    res = RagService().chain.invoke({"input": "针织毛衣如何保养？"}, session_config)
    print(res)

