from core.genai_client import init_genai_client
from core.config import settings

def generate_answer(prompt: str) -> str:
    client = init_genai_client()
    resp = client.models.generate_content(model=settings.GEN_MODEL, contents=prompt)
    return resp.text