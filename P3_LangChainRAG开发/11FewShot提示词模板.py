"""LangChain 通过 OpenAI 兼容接口调用火山方舟 Coding Plan，演示 Few-Shot 提示词模板。"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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

# 示例的模板
example_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")

# 示例的动态数据注入 要求是list内部套字典
examples_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"},
]

few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,  # 示例数据的模板
    examples=examples_data,  # 示例的数据（用来注入动态数据的），list内套字典
    prefix="告知我单词的反义词，我提供如下的示例：",  # 示例之前的提示词
    suffix="基于前面的示例告知我，{input_word}的反义词是？",  # 示例之后的提示词
    input_variables=['input_word']  # 声明在前缀或后缀中所需要注入的变量名
)

prompt_text = few_shot_template.invoke(input={"input_word": "左"}).to_string()
print(prompt_text)

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
