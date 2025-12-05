from core.db import get_db_conn
import json

def save_log(question, answer, username, chunks_used, scores, metadata=None):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO rag_logs (question, answer, username, chunks_used, scores, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, [
            question,
            answer,
            username,
            json.dumps(chunks_used, ensure_ascii=False),
            json.dumps(scores),
            json.dumps(metadata or {})
        ])
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("Error logging RAG interaction:", e)