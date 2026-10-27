
md5_path = "./md5.text"


# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000        # 文本分割的阈值

#相似度检索阈值
similarity_threshold = 2            # 检索返回匹配的文档数量

embedding_model_name = "doubao-embedding-vision"
chat_model_name = "deepseek-v4-flash"

session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
