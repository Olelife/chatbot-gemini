import logging
from fastapi import APIRouter, HTTPException, Query
from rag.knowledge import build_or_load_knowledge_base
from rag.cache import (
    load_cache_local, save_cache_local,
    upload_cache_to_gcs, download_cache_from_gcs
)
from rag.loader import list_files_in_folder, load_file_as_units
from rag.embeddings import embed_texts
from core.config import settings
import os
import hashlib
import numpy as np

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = logging.getLogger(__name__)


@router.post("/rebuild-kb")
def rebuild_kb(
    country: str = Query(..., description="País: mx, br"),
    model: str = Query("text-embedding-005", description="Modelo de embeddings"),
):
    """
    Reconstruye la base de conocimiento *completa* para el país indicado
    usando el modelo de embeddings indicado.
    """

    country = country.lower()
    cache_local = f"/tmp/kb_cache_{country}.pkl"
    cache_gcs = f"gemini-ai/cache/kb_cache_{country}.pkl"

    try:
        # ============================================
        # 1. BORRAR CACHE LOCAL
        # ============================================
        if os.path.exists(cache_local):
            os.remove(cache_local)
            logger.info(f"[ADMIN] Local cache deleted → {cache_local}")

        # ============================================
        # 2. BORRAR CACHE EN GCS
        # ============================================
        from google.cloud import storage
        client = storage.Client()
        bucket_obj = client.bucket(settings.BUCKET)
        blob = bucket_obj.blob(cache_gcs)

        if blob.exists():
            blob.delete()
            logger.info(f"[ADMIN] Remote cache deleted → gs://{settings.BUCKET}/{cache_gcs}")

        # ============================================
        # 3. CARGAR TODOS LOS ARCHIVOS RAW
        # ============================================
        folder = f"{settings.KNOWLEDGE_FOLDER}/{country}"
        files = list_files_in_folder(settings.BUCKET, folder)

        units = []
        for f in files:
            units.extend(load_file_as_units(settings.BUCKET, f))

        logger.info(f"[ADMIN] {country} → {len(units)} semantic units")

        # ============================================
        # 4. GENERAR NUEVOS EMBEDDINGS
        # ============================================
        embeddings = embed_texts(units, model=model)

        dim = embeddings.shape[1]

        logger.info(f"[ADMIN] Embeddings generated → dim={dim}")

        # ============================================
        # 5. GUARDAR CACHE LOCAL + GCS
        # ============================================
        save_cache_local(units, embeddings, cache_local)
        upload_cache_to_gcs(settings.BUCKET, cache_gcs, cache_local)

        # ============================================
        # 6. RETORNAR RESPUESTA
        # ============================================
        return {
            "status": "ok",
            "country": country,
            "model": model,
            "chunks": len(units),
            "embedding_dim": dim,
            "cache_local": cache_local,
            "cache_gcs": cache_gcs,
        }

    except Exception as e:
        logger.error(f"[ADMIN] Rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
