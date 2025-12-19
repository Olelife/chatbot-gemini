import redis
import json
from core.config import settings

# Conexión Redis (ajusta para Memorystore o local)
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True  # retorna strings en vez de bytes
)

SESSION_TTL = 7200  # 2 horas


def get_session_key(session_id: str, country: str) -> str:
    return f"chat:{country}:{session_id}"


def get_history(session_id: str, country: str):
    key = get_session_key(session_id, country)
    data = redis_client.get(key)
    if not data:
        return []
    return json.loads(data)


def save_history(session_id: str, country: str, history):
    key = get_session_key(session_id, country)
    redis_client.set(key, json.dumps(history), ex=SESSION_TTL)
