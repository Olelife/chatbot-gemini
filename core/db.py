import psycopg2
import psycopg2.extras

from core.config import settings

def get_db_conn():
    return psycopg2.connect(
        host=settings.API_CHAT_GEMINI_DB_HOST,
        user=settings.API_CHAT_GEMINI_DB_USER,
        password=settings.API_CHAT_GEMINI_DB_PASS,
        database=settings.API_CHAT_GEMINI_DB_NAME,
        cursor_factory=psycopg2.extras.RealDictCursor
    )