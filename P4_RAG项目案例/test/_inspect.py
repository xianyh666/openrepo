import sqlite3

con = sqlite3.connect(r"c:\Users\yh\PycharmProjects\LangChain\P4_RAG项目案例\chroma_db\chroma.sqlite3")
cur = con.cursor()

print("== embeddings ==")
for r in cur.execute("select * from embeddings"):
    print(r)

print("\n== embedding_metadata ==")
for r in cur.execute("select * from embedding_metadata"):
    print(r)

print("\n== embedding_fulltext_search (rowid + string_value) ==")
for r in cur.execute("select rowid, string_value from embedding_fulltext_search"):
    print(r)
