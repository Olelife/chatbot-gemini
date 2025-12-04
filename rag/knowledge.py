# app/rag/knowledge.py
import os
import json
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from google.cloud import storage

from core.config import settings
from rag.embeddings import embed_texts
from rag.cache import (
    load_cache_local,
    save_cache_local,
    download_cache_from_gcs,
    upload_cache_to_gcs,
    clear_cache_local,
    clear_cache_gcs,
)
import logging

logger = logging.getLogger(__name__)

# KB en memoria
kb_chunks: Optional[List[str]] = None
kb_vectors: Optional[np.ndarray] = None
kb_metadata: Optional[Dict[str, Any]] = None


def _storage_client():
    return storage.Client()


def list_files() -> List[str]:
    """
    Lista todos los archivos en el folder de knowledge.
    """
    client = _storage_client()
    bucket_obj = client.bucket(settings.BUCKET)

    files = [
        blob.name
        for blob in bucket_obj.list_blobs(prefix=settings.KNOWLEDGE_FOLDER)
        if not blob.name.endswith("/")
    ]

    # Excluir prompts si existen
    return [f for f in files if "prompts" not in f]


def _load_file_raw(path: str) -> bytes:
    client = _storage_client()
    bucket_obj = client.bucket(settings.BUCKET)
    blob = bucket_obj.blob(path)

    if not blob.exists():
        logger.warning(f"File not found in GCS: {path}")
        return b""

    return blob.download_as_bytes()


def load_file_as_units(path: str) -> List[str]:
    """
    Carga un archivo JSON o texto y lo convierte en chunks semánticos completos.
    - Coberturas: un chunk por cobertura.
    - Diccionarios: un chunk por archivo.
    - Listas: un chunk por elemento.
    """
    raw = _load_file_raw(path)
    if not raw:
        return []

    # No JSON → chunk único de texto
    if not path.endswith(".json"):
        text = raw.decode("utf-8", errors="ignore")
        return [text] if text.strip() else []

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        logger.error(f"Error parsing JSON {path}: {e}")
        return []

    units: List[str] = []

    # LISTA
    if isinstance(data, list):
        for entry in data:
            nombre = (
                entry.get("nombre")
                or entry.get("titulo")
                or entry.get("id")
                or "ITEM"
            )
            block = (
                f"## ITEM: {nombre}\n\n"
                f"{json.dumps(entry, ensure_ascii=False, indent=2)}"
            )
            units.append(block)
        return units

    # DICCIONARIO
    if isinstance(data, dict):
        # Detectar archivos de coberturas por ruta o campos típicos
        if "coberturas" in path or "cobertura" in path:
            nombre = (
                data.get("nombre")
                or data.get("id")
                or os.path.basename(path).replace(".json", "")
            )
            block = (
                f"## COBERTURA: {nombre}\n\n"
                f"{json.dumps(data, ensure_ascii=False, indent=2)}"
            )
            return [block]

        # Archivos tipo FAQ, operativa, rules, lists, etc.
        nombre = os.path.basename(path).replace(".json", "").upper()
        block = (
            f"## SECCIÓN: {nombre}\n\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)}"
        )
        return [block]

    # Último recurso
    return [json.dumps(data, ensure_ascii=False)]


def build_knowledge_units_from_folder() -> List[str]:
    files = list_files()
    all_units: List[str] = []

    for f in files:
        units = load_file_as_units(f)
        all_units.extend(units)

    logger.info(f"Knowledge units created: {len(all_units)}")
    return all_units


def _build_from_raw() -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    from rag.faiss_index import build_faiss_index
    """
    Construye la KB desde los JSON del bucket.
    """
    logger.info("Building KB from raw knowledge folder...")
    chunks = build_knowledge_units_from_folder()
    vectors = embed_texts(chunks)

    # guardar cache
    save_cache_local(chunks, vectors)
    upload_cache_to_gcs()

    # Construir índice FAISS
    build_faiss_index(vectors, use_hnsw=True)

    # obtener metadata desde el cache recien guardado
    c2, v2, meta = load_cache_local()
    if c2 is None or v2 is None:
        # fallback: reconstruir metadata simple
        meta = {
            "chunks_count": len(chunks),
            "embedding_dim": int(vectors.shape[1]),
        }
        return chunks, vectors, meta

    return c2, v2, meta


def get_kb() -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    from rag.faiss_index import build_faiss_index
    """
    Devuelve (chunks, vectors, metadata), cargando desde:
    - memoria (si ya está)
    - cache local
    - cache en GCS
    - o reconstruyendo desde knowledge/
    """
    global kb_chunks, kb_vectors, kb_metadata

    if kb_chunks is not None and kb_vectors is not None:
        return kb_chunks, kb_vectors, kb_metadata or {}

    # 1) Intentar cache local
    c, e, m = load_cache_local()
    if c is not None and e is not None:
        logger.info("Loaded KB from local cache")
        kb_chunks, kb_vectors, kb_metadata = c, e, m
        build_faiss_index(e, use_hnsw=True)
        return kb_chunks, kb_vectors, kb_metadata

    # 2) Intentar cache GCS
    if download_cache_from_gcs():
        c, e, m = load_cache_local()
        if c is not None and e is not None:
            logger.info("Loaded KB from GCS cache")
            kb_chunks, kb_vectors, kb_metadata = c, e, m
            build_faiss_index(e, use_hnsw=True)
            return kb_chunks, kb_vectors, kb_metadata

    # 3) Construir desde knowledge/
    kb_chunks, kb_vectors, kb_metadata = _build_from_raw()
    return kb_chunks, kb_vectors, kb_metadata or {}


def rebuild_knowledge() -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Borra caches y reconstruye la KB desde cero.
    """
    global kb_chunks, kb_vectors, kb_metadata

    clear_cache_local()
    clear_cache_gcs()

    kb_chunks, kb_vectors, kb_metadata = _build_from_raw()
    return kb_chunks, kb_vectors, kb_metadata
