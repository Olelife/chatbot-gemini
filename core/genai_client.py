from typing import Optional

from google import genai
from core.config import settings
import logging

logger = logging.getLogger(__name__)

genai_client: Optional[genai.Client] = None

def ensure_genai_client():
    global genai_client
    if genai_client is None:
        logger.info("Initializing GenAI Vertex client...")
        genai_client = genai.Client(
            vertexai=True,
            project=settings.PROJECT_ID,
            location=settings.LOCATION
        )
    return genai_client