import os
import logging
from fastapi import APIRouter
from core.config import settings
from rag.knowledge import build_or_load_knowledge_base
from google.cloud import storage

router = APIRouter(prefix="/reload", tags=["Ask"])
logger = logging.getLogger(__name__)

@router.post("kb")
def reload_kb():
    """
    Fuerza la recarga completa de la base de conocimiento:
    - Borra cache local
    - Borra cache en GCS
    - Reconstruye chunks + embeddings
    """
    global chunks, chunk_embeddings, kb_metadata

    # 1. Borrar cache local
    if os.path.exists(settings.CACHE_LOCAL_PATH):
        os.remove(settings.CACHE_LOCAL_PATH)

    # 2. Borrar cache en GCS
    try:
        client = storage.Client()
        bucket_obj = client.bucket(settings.BUCKET)
        blob = bucket_obj.blob(settings.CACHE_GCS_PATH)
        if blob.exists():
            blob.delete()
    except Exception as e:
        return {"error": f"Could not delete GCS cache: {e}"}

    # 3. Reconstruir KB desde cero
    try:
        chunks, chunk_embeddings, kb_metadata = build_or_load_knowledge_base()
    except Exception as e:
        return {"error": f"Failed to rebuild KB: {e}"}

    return {
        "status": "reloaded",
        "chunks_count": len(chunks),
        "embedding_dim": int(chunk_embeddings.shape[1]),
        "metadata": kb_metadata,
    }