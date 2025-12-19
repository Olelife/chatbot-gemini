# rag/session_memory.py
from typing import List, Dict, Tuple

# Estructura:
# SESSIONS[(session_id, country)] = [ { "role": "user"/"assistant", "content": "..." }, ... ]
SESSIONS: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

MAX_TURNS = 6  # cantidad máxima de mensajes (user+assistant) que guardamos por sesión/país


def get_history(session_id: str, country: str) -> List[Dict[str, str]]:
    """
    Devuelve el historial de una sesión para un país específico.
    """
    key = (session_id, country)
    return SESSIONS.get(key, [])


def add_to_history(session_id: str, country: str, role: str, content: str) -> None:
    """
    Agrega un mensaje al historial (por sesión y país).
    role: "user" o "assistant"
    """
    key = (session_id, country)

    if key not in SESSIONS:
        SESSIONS[key] = []

    SESSIONS[key].append({"role": role, "content": content})

    # Limitamos el tamaño del historial
    if len(SESSIONS[key]) > MAX_TURNS:
        SESSIONS[key] = SESSIONS[key][-MAX_TURNS:]


def build_history_text(session_id: str, country: str) -> str:
    """
    Construye un texto compacto con el historial para usar en el prompt.
    """
    history = get_history(session_id, country)

    if not history:
        return ""

    lines = []
    for turn in history:
        speaker = "USUARIO" if turn["role"] == "user" else "ASISTENTE"
        lines.append(f"{speaker}: {turn['content']}")

    return "\n".join(lines)
