# app/core/genai_client.py
from google import genai
from core.config import settings
import logging

logger = logging.getLogger(__name__)

_client = None  # privado


def init_genai_client():
    global _client
    if _client is None:
        logger.info("Initializing GenAI client...")
        _client = genai.Client(
            vertexai=True,
            project=settings.PROJECT_ID,
            location=settings.LOCATION,
        )
        logger.info("GenAI client initialized")
    return _client


def get_client():
    if _client is None:
        raise RuntimeError("GenAI client not initialized")
    return _client