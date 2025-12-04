# app/rag/embeddings.py
from core.genai_client import get_client
from core.config import settings
from google.genai import types
import numpy as np
import logging

logger = logging.getLogger(__name__)


def embed_texts(texts):
    client = get_client()  # ← cliente siempre actualizado
    resp = client.models.embed_content(
        model=settings.EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return np.vstack([np.array(e.values, dtype="float32") for e in resp.embeddings])


def embed_single(text: str):
    return embed_texts([text])[0]
