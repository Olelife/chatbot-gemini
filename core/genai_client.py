import logging
from typing import Optional
from google import genai
from core.config import settings

logger = logging.getLogger(__name__)

_genai_client: Optional[genai.Client] = None


def init_genai_client():
    """
    Inicializa el cliente global solo una vez.
    """
    global _genai_client
    if _genai_client is None:
        logger.info("Initializing GenAI Vertex client…")
        _genai_client = genai.Client(
            vertexai=True,
            project=settings.PROJECT_ID,
            location=settings.LOCATION
        )
    return _genai_client


def get_client() -> genai.Client:
    """
    Obtiene un cliente ya inicializado.
    Si no está inicializado, lo inicializa automáticamente.
    """
    global _genai_client
    if _genai_client is None:
        return init_genai_client()
    return _genai_client
