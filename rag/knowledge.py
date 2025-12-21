import logging
from typing import List, Tuple
import hashlib
import numpy as np

from core.config import settings
from integrations.bigquery_zendesk import fetch_zendesk_articles_br
from rag.cache import save_cache_local, load_cache_local, upload_cache_to_gcs, download_cache_from_gcs
from rag.loader import load_file_as_units, list_files_in_folder
from rag.embeddings import embed_texts
from rag.zendesk_chunks import convert_zendesk_articles_to_chunks

logger = logging.getLogger(__name__)


def build_or_load_knowledge_base(country: str) -> Tuple[List[str], np.ndarray, dict]:
    """
    Construye o carga la KB específica de un país.
    country: "mx", "br", etc.
    Incluye datos externos (Zendesk) solo cuando aplica.
    """

    # Cache por país
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

    json_units = []
    for f in files:
        json_units.extend(load_file_as_units(settings.BUCKET, f))

    logger.info(f"[KB] JSON chunks loaded for {country}: {len(json_units)}")

    json_embeddings = embed_texts(json_units)

    zendesk_units = []
    zendesk_embeddings = None

    if country == "br":
        logger.info("[KB] Fetching Zendesk BR articles from BigQuery...")

        articles = fetch_zendesk_articles_br()
        zendesk_units = convert_zendesk_articles_to_chunks(articles)

        logger.info(f"[KB] Zendesk BR chunks: {len(zendesk_units)}")

        if len(zendesk_units) > 0:
            zendesk_embeddings = embed_texts(zendesk_units)

    if zendesk_units:
        all_chunks = json_units + zendesk_units
        all_embeddings = np.vstack([json_embeddings, zendesk_embeddings])
    else:
        all_chunks = json_units
        all_embeddings = json_embeddings

    metadata = {
        "json_chunks": len(json_units),
        "zendesk_chunks": len(zendesk_units),
        "total_chunks": len(all_chunks),
        "embedding_dim": all_embeddings.shape[1],
        "hash": hashlib.sha256("\n".join(all_chunks).encode()).hexdigest(),
    }

    # Guardar cache local
    save_cache_local(all_chunks, all_embeddings, local_cache_path)
    # Subir cache a GCS
    upload_cache_to_gcs(settings.BUCKET, gcs_cache_path, local_cache_path)

    logger.info(f"[KB] Built KB for {country} → total chunks: {len(all_chunks)}")

    return all_chunks, all_embeddings, metadata