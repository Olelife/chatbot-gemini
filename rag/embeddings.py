import numpy as np
from google.genai import types
from core.genai_client import get_client
from core.config import settings

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeddings multi-país.
    Separamos por país para evitar mezclar KB de México con Brasil.
    """

    client = get_client()

    response = client.models.embed_content(
        model=settings.EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            # IMPORTANTÍSIMO: incluir país en el namespace
            # para evitar contaminación entre regiones
            output_dimensionality=None,
        ),
    )

    vectors = [np.array(e.values, dtype="float32") for e in response.embeddings]
    return np.vstack(vectors)

def embed_single(text: str) -> np.ndarray:
    return embed_texts([text])[0]

