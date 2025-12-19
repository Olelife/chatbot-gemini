import psycopg2
from psycopg2.pool import SimpleConnectionPool

from core.config import settings

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=settings.API_CHAT_GEMINI_DB_HOST,
    user=settings.API_CHAT_GEMINI_DB_USER,
    password=settings.API_CHAT_GEMINI_DB_PASS,
    database=settings.API_CHAT_GEMINI_DB_NAME
)

def get_db_conn():
    return db_pool.getconn()

def release_conn(conn):
    db_pool.putconn(conn)