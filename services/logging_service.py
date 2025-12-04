from core.db import get_conn
import json

def save_log(question, answer, username, chunks_used, scores, metadata):
    conn = get_conn()
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
        json.dumps(metadata)
    ])

    conn.commit()
    cur.close()
    conn.close()