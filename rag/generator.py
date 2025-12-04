# app/rag/generator.py
from core.genai_client import get_client
from core.config import settings
from google.genai import types


def choose_model(question: str, context: str) -> str:
    """Selecciona automáticamente entre Flash y Pro."""

    # Si la pregunta es muy corta → usa Flash (mucho más rápido)
    if len(question) < 60:
        return "gemini-2.5-pro"

    # Si el contexto es enorme → usar Pro
    if len(context) > 2500:
        return settings.GEN_MODEL  # gemini-2.5-pro

    # Si el usuario está en saludo o cosas triviales
    trivial_words = ["hola", "hi", "saludos", "nombre", "?" ]
    if any(w in question.lower() for w in trivial_words):
        return "gemini-2.5-pro"

    # Default
    return settings.GEN_MODEL


def generate_answer(prompt: str, question: str, context: str) -> str:
    client = get_client()
    model_name = choose_model(question, context)

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=800,
        )
    )
    return resp.text
