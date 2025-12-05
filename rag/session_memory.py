from typing import List, Dict

SESSIONS = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    return SESSIONS.get(session_id, [])

def add_to_history(session_id: str, role: str, content: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    SESSIONS[session_id].append({"role": role, "content": content})

    # Limitar a las últimas 5 interacciones
    SESSIONS[session_id] = SESSIONS[session_id][-5:]

def build_history_text(session_id: str) -> str:
    history = get_history(session_id)

    formatted = ""
    for turn in history:
        formatted += f"{turn['role'].upper()}: {turn['content']}\n"

    return formatted.strip()