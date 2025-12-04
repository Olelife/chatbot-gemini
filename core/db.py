import psycopg2
import psycopg2.extras
from core.config import settings

def get_conn():
    return psycopg2.connect(
        host=settings.DB_HOST,
        user=settings.DB_USER,
        password=settings.DB_PASS,
        dbname=settings.DB_NAME,
        cursor_factory=psycopg2.extras.RealDictCursor
    )
