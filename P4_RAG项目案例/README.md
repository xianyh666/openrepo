# P4_RAG 项目案例 — 前端启动指南

本项目使用 **Streamlit** 作为前端框架，包含两个前端应用：

| 前端 | 文件 | 功能 |
|------|------|------|
| 知识库更新服务 | `app_file_uploader.py` | 网页上传 TXT 文件，向量化后存入 Chroma 向量库 |
| 智能客服 QA | `app_qa.py` | 基于向量库检索 + 大模型生成回答的聊天问答页 |

## 一、前置条件

| 工具 | 要求 | 说明 |
|------|------|------|
| Python | >= 3.14 | 项目依赖由 uv 管理 |
| uv | 任意较新版本 | 推荐使用 uv 管理依赖与运行环境 |
| 火山方舟账号 | 已开通 API | 需要可用的 `ARK_API_KEY` 与 `ARK_BASE_URL` |

## 二、首次运行：安装依赖

在项目**根目录**（`LangChain`，即包含 `pyproject.toml` 的目录）执行：

```powershell
cd "C:\Users\yh\PycharmProjects\LangChain"
uv sync
```

> 若 `uv` 未安装，可先执行 `pip install uv` 或参考官网安装。

## 三、配置环境变量

项目根目录（`LangChain`）下需要存在 `.env` 文件，内容如下：

```env
ARK_API_KEY=你的火山方舟 Coding Plan API Key
ARK_BASE_URL=你的 Coding Plan Base URL
```

配置完成后启动前端即可正常调用 Embedding 模型。

## 四、启动前端

### 方式一：启动"知识库更新服务"（上传 TXT 入库）

```powershell
cd "C:\Users\yh\PycharmProjects\LangChain\P4_RAG项目案例"
uv run streamlit run app_file_uploader.py
```

### 方式二：启动"智能客服 QA"（聊天问答）

```powershell
cd "C:\Users\yh\PycharmProjects\LangChain\P4_RAG项目案例"
uv run streamlit run app_qa.py
```

- `uv run` 会自动使用根目录的 `.venv` 环境，无需手动激活虚拟环境。
- 启动成功后，浏览器会自动打开，也可手动访问 **http://localhost:8501**。
- 若 8501 端口被占用，Streamlit 会自动换端口（日志中会显示实际地址）。
- **两个前端同时运行必须分开端口**，例如：

```powershell
uv run streamlit run app_file_uploader.py --server.port 8501
uv run streamlit run app_qa.py --server.port 8502
```

> 注意：`app_qa.py` 启动时会初始化检索链（连接向量库 + 创建模型），首次打开页面需要等待几秒属正常现象。

## 五、页面使用说明

### 知识库更新服务（`app_file_uploader.py`）

1. 页面标题为 **"知识库更新服务"**。
2. 点击"请上传 TXT 文件"，选择一个 `.txt` 文件（仅支持单文件上传）。
3. 页面会展示文件名、格式与大小。
4. 上传后自动完成 MD5 去重校验 → 文本分割 → 向量化 → 写入 Chroma 向量库。
5. 结果提示：
   - `[跳过]内容已经存在知识库中`：该文件已处理过，无需重复入库。
   - `[成功]内容已经成功载入向量库`：入库成功。

### 智能客服 QA（`app_qa.py`）

1. 页面标题为 **"智能客服"**，默认有一句欢迎语。
2. 在页面最下方输入框提问，例如"针织毛衣如何保养？"。
3. 回答流程：问题向量化 → 检索向量库中最相近的片段 → 拼入 prompt → 调用大模型流式生成回答。
4. 对话历史自动保存在 `chat_history/<session_id>` 文件中，多轮对话可引用上下文。

## 六、常见问题

| 问题 | 解决方案 |
|------|----------|
| 提示 `未找到 ARK_API_KEY` | 检查根目录 `.env` 文件是否存在且已填写正确的 Key |
| `uv run` 找不到项目 | 请确认在 `P4_RAG项目案例` 或 `LangChain` 目录下执行（uv 会向上查找 `pyproject.toml`） |
| 端口 8501 被占用 | 追加参数指定端口：`uv run streamlit run app_qa.py --server.port 8502`（或任意未占用端口） |
| `app_qa.py` 页面加载很久 | 首次启动会初始化检索链（连接 Chroma + 创建模型），属正常现象，稍等即可 |
| 上传报编码错误 | 请确保上传文件为 UTF-8 编码的 TXT 文件 |
