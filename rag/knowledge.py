from core.genai_client import ensure_genai_client
from rag.cache import load_cache_local, download_cache_from_gcs, save_cache_local, upload_cache_to_gcs
from utils.file import list_files_in_folder
from google.cloud import storage
from typing import Optional, Tuple, List
import os
import logging
from core.config import settings
import numpy as np
from google.genai import types

logger = logging.getLogger(__name__)

def load_file_as_units(bucket: str, path: str) -> List[str]:
    """
    Carga un archivo JSON y lo convierte en chunks semánticos completos.
    Las coberturas se agrupan en un solo chunk.
    Diccionarios se mantienen como una unidad completa.
    Listas generan un chunk por elemento.
    No se divide por claves individuales.
    """

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(path)

    if not blob.exists():
        return []

    raw = blob.download_as_bytes()

    # Archivos que no son JSON → chunk único
    if not path.endswith(".json"):
        text = raw.decode("utf-8", errors="ignore")
        return [text] if text.strip() else []

    import json
    try:
        data = json.loads(raw.decode("utf-8"))
    except:
        return []

    units = []

    # -----------------------------
    # CASO 1: JSON es LISTA
    # -----------------------------
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

    # -----------------------------
    # CASO 2: JSON es DICCIONARIO
    # -----------------------------
    if isinstance(data, dict):

        # Detectar archivos de COBERTURAS (muy importante)
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

        # Archivos tipo secciones (faq, operativa, lists, etc.)
        nombre = os.path.basename(path).replace(".json", "").upper()

        block = (
            f"## SECCIÓN: {nombre}\n\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)}"
        )
        return [block]

    # Último recurso: chunk único
    return [json.dumps(data)]

def build_knowledge_units_from_folder(bucket: str, root_folder: str) -> List[str]:
    """
    Construye chunks semánticos desde múltiples archivos JSON/texto
    dentro del folder knowledge/.
    """
    files = list_files_in_folder(bucket, root_folder)
    all_units = []

    for f in files:
        if "prompts" in f:
            continue

        units = load_file_as_units(bucket, f)
        all_units.extend(units)

    logger.info(f"Knowledge units created: {len(all_units)}")
    return all_units

def embed_texts(texts: List[str]) -> np.ndarray:
    client = ensure_genai_client()

    response = client.models.embed_content(
        model=settings.EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    vectors = [np.array(e.values, dtype="float32") for e in response.embeddings]
    return np.vstack(vectors)

def embed_single(text: str) -> np.ndarray:
    return embed_texts([text])[0]

def build_or_load_knowledge_base():
    # 1) Intentar cache local
    local_chunks, local_emb, metadata = load_cache_local()
    if local_chunks is not None:
        logger.info("Loaded KB from LOCAL CACHE")
        return local_chunks, local_emb, metadata

    # 2) Intentar cache GCS
    if download_cache_from_gcs(settings.BUCKET, settings.CACHE_GCS_PATH):
        c, e, m = load_cache_local()
        if c is not None:
            logger.info("Loaded KB from GCS CACHE")
            return c, e, m

    # 3) Construir desde knowledge/
    logger.info("Building KB from raw JSON knowledge folder…")

    kb_chunks = build_knowledge_units_from_folder(settings.BUCKET, settings.KNOWLEDGE_FOLDER)
    logger.info(f"KB semantic chunks: {len(kb_chunks)}")
    kb_embeddings = embed_texts(kb_chunks)

    save_cache_local(kb_chunks, kb_embeddings)
    upload_cache_to_gcs(settings.BUCKET, settings.CACHE_GCS_PATH)

    # cargar metadata recien guardada
    _, _, metadata = load_cache_local()

    return kb_chunks, kb_embeddings, metadata