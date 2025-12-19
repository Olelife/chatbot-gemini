import logging
from typing import List, Tuple
import hashlib
import numpy as np

from core.config import settings
from rag.cache import save_cache_local, load_cache_local, upload_cache_to_gcs, download_cache_from_gcs
from rag.loader import load_file_as_units, list_files_in_folder
from rag.embeddings import embed_texts

logger = logging.getLogger(__name__)


def build_or_load_knowledge_base(country: str) -> Tuple[List[str], np.ndarray, dict]:
    """
    Construye o carga la KB específica de un país.
    country: "mx", "br", "cl", "pe", etc.
    """

    local_cache_path = f"/tmp/kb_cache_{country}.pkl"
    gcs_cache_path = f"gemini-ai/cache/kb_cache_{country}.pkl"

    # 1. Intentar cache local
    chunks, embeddings, metadata = load_cache_local(local_cache_path)
    if chunks is not None:
        logger.info(f"Loaded KB for {country} from local cache")
        return chunks, embeddings, metadata

    # 2. Intentar cache GCS
    if download_cache_from_gcs(settings.BUCKET, gcs_cache_path, local_cache_path):
        chunks, embeddings, metadata = load_cache_local(local_cache_path)
        if chunks is not None:
            logger.info(f"Loaded KB for {country} from GCS cache")
            return chunks, embeddings, metadata

    # 3. Construir KB desde cero
    folder = f"{settings.KNOWLEDGE_FOLDER}/{country}"
    files = list_files_in_folder(settings.BUCKET, folder)

    units = []
    for f in files:
        units.extend(load_file_as_units(settings.BUCKET, f))

    logger.info(f"KB for {country}: {len(units)} chunks")

    embeddings = embed_texts(units)

    # Guardar cache local
    save_cache_local(units, embeddings, local_cache_path)
    # Subir cache a GCS
    upload_cache_to_gcs(settings.BUCKET, gcs_cache_path, local_cache_path)

    metadata = {
        "chunks": len(units),
        "embedding_dim": embeddings.shape[1],
        "hash": hashlib.sha256("\n".join(units).encode()).hexdigest()
    }

    return units, embeddings, metadata
