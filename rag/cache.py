# app/rag/cache.py
import os
import pickle
import time
import hashlib
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from google.cloud import storage

from core.config import settings
import logging

logger = logging.getLogger(__name__)


def _compute_metadata(chunks: List[str], embeddings: np.ndarray) -> Dict[str, Any]:
    return {
        "created_at": time.time(),
        "chunks_count": len(chunks),
        "embedding_dim": embeddings.shape[1] if embeddings.size else 0,
        "hash": hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest(),
    }


def save_cache_local(chunks: List[str], embeddings: np.ndarray) -> None:
    try:
        metadata = _compute_metadata(chunks, embeddings)
        with open(settings.CACHE_LOCAL_PATH, "wb") as f:
            pickle.dump(
                {
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "metadata": metadata,
                },
                f,
            )
        logger.info("Local cache saved")
    except Exception as e:
        logger.warning(f"Could not save local cache: {e}")


def load_cache_local() -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[dict]]:
    if not os.path.exists(settings.CACHE_LOCAL_PATH):
        return None, None, None

    try:
        with open(settings.CACHE_LOCAL_PATH, "rb") as f:
            data = pickle.load(f)
        logger.info("Local cache loaded")
        return data.get("chunks"), data.get("embeddings"), data.get("metadata")
    except Exception as e:
        logger.warning(f"Could not load local cache: {e}")
        return None, None, None


def upload_cache_to_gcs() -> None:
    try:
        client = storage.Client()
        bucket_obj = client.bucket(settings.BUCKET)
        blob = bucket_obj.blob(settings.CACHE_GCS_PATH)
        blob.upload_from_filename(settings.CACHE_LOCAL_PATH)
        logger.info("Cache uploaded to GCS")
    except Exception as e:
        logger.warning(f"Could not upload GCS cache: {e}")


def download_cache_from_gcs() -> bool:
    try:
        client = storage.Client()
        bucket_obj = client.bucket(settings.BUCKET)
        blob = bucket_obj.blob(settings.CACHE_GCS_PATH)

        if not blob.exists():
            return False

        blob.download_to_filename(settings.CACHE_LOCAL_PATH)
        logger.info("Cache downloaded from GCS")
        return True
    except Exception as e:
        logger.warning(f"Error downloading cache from GCS: {e}")
        return False


def clear_cache_local() -> None:
    if os.path.exists(settings.CACHE_LOCAL_PATH):
        os.remove(settings.CACHE_LOCAL_PATH)
        logger.info("Local cache removed")


def clear_cache_gcs() -> None:
    try:
        client = storage.Client()
        bucket_obj = client.bucket(settings.BUCKET)
        blob = bucket_obj.blob(settings.CACHE_GCS_PATH)
        if blob.exists():
            blob.delete()
            logger.info("GCS cache removed")
    except Exception as e:
        logger.warning(f"Could not delete GCS cache: {e}")
