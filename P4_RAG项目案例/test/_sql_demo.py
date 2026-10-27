import sqlite3
from pathlib import Path

if hasattr(__import__("sys").stdout, "reconfigure"):
    __import__("sys").stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[1] / "chroma_db" / "chroma.sqlite3"
con = sqlite3.connect(str(DB_PATH))
cur = con.cursor()

print("=== 1. 查看所有表 ===")
for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(r[0])

print("\n=== 2. 查看表结构(以 embeddings 为例) ===")
for r in cur.execute("PRAGMA table_info(embeddings)"):
    print(r)

print("\n=== 3. 查看集合 collections ===")
for r in cur.execute("SELECT * FROM collections"):
    print(r)

print("\n=== 4. 文本内容(chroma:document) ===")
for r in cur.execute("SELECT string_value FROM embedding_metadata WHERE key='chroma:document'"):
    print(r[0])

print("\n=== 5. 全部元数据 ===")
for r in cur.execute("SELECT id, key, string_value FROM embedding_metadata"):
    print(r)

con.close()
