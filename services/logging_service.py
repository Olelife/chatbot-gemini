from core.db import get_db_conn, release_conn
import json
import logging

logger = logging.getLogger(__name__)

def save_log_async(question, answer, username, chunks_used, scores, metadata=None):
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
    except Exception as e:
        logger.error(f"DB Log Error: {e}")
    finally:
        release_conn(conn)