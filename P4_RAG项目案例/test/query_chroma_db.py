"""查看 chroma_db 中存储了哪些数据的脚本

纯 Python 标准库实现, 不依赖 langchain / dotenv / API Key,
任何 Python 3 环境都能直接运行。

用法:
    python test/query_chroma_db.py
"""
import sqlite3
import sys
from pathlib import Path

# Windows 控制台默认 GBK, 重新配置为标准 UTF-8, 避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 数据库文件: 本文件(test/) 的上一级目录 -> chroma_db/chroma.sqlite3
DB_PATH = Path(__file__).resolve().parents[1] / "chroma_db" / "chroma.sqlite3"

if not DB_PATH.exists():
    print(f"[错误] 未找到数据库文件: {DB_PATH}")
    sys.exit(1)

con = sqlite3.connect(str(DB_PATH))
cur = con.cursor()

# 1. 集合信息
collections = cur.execute(
    "SELECT id, name, dimension FROM collections"
).fetchall()
if not collections:
    print("数据库中没有集合(collection)。")
    sys.exit(0)

for c in collections:
    print(f"collection: {c[1]}  (向量维度={c[2]})")

# 2. 每条向量记录(通过 segments 关联到集合)
records = cur.execute(
    """
    SELECT e.id AS row_id, e.embedding_id, s.collection
    FROM embeddings e
    JOIN segments s ON s.id = e.segment_id
    """
).fetchall()

print(f"总条数: {len(records)}")
print("=" * 60)

if not records:
    print("数据库中没有向量数据。")
    sys.exit(0)

# 3. 元数据(每条记录的多行键值对 -> 合并成字典)
meta_rows = cur.execute(
    "SELECT id, key, string_value FROM embedding_metadata"
).fetchall()
meta_map = {}
for r in meta_rows:
    meta_map.setdefault(r[0], []).append(r)

for i, rec in enumerate(records):
    metas = meta_map.get(rec[0], [])
    doc = ""
    meta_dict = {}
    for m in metas:
        if m[1] == "chroma:document":
            doc = m[2] or ""
        else:
            meta_dict[m[1]] = m[2]

    print(f"\n[{i}] id={rec[1]}")
    print(f"    文本: {doc[:100]}...")
    print(f"    元数据: {meta_dict}")

con.close()
