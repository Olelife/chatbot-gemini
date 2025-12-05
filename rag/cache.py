import os
import time
import logging
from core.config import settings
import hashlib
import pickle
from google.cloud import storage

logger = logging.getLogger(__name__)

def save_cache_local(chunks_list, embeddings):
    try:
        metadata = {
            "created_at": time.time(),
            "chunks_count": len(chunks_list),
            "embedding_dim": embeddings.shape[1] if embeddings.size else 0,
            "hash": hashlib.sha256("\n".join(chunks_list).encode("utf-8")).hexdigest(),
        }

        with open(settings.CACHE_LOCAL_PATH, "wb") as f:
            pickle.dump(
                {
                    "chunks": chunks_list,
                    "embeddings": embeddings,
                    "metadata": metadata,
                },
                f,
            )
        logger.info("Local cache saved")
    except Exception as e:
        logger.warning(f"Could not save local cache: {e}")

def load_cache_local():
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

def upload_cache_to_gcs(bucket: str, path: str):
    try:
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(path)
        blob.upload_from_filename(settings.CACHE_LOCAL_PATH)
        logger.info("Cache uploaded to GCS")
    except Exception as e:
        logger.warning(f"Could not upload GCS cache: {e}")


def download_cache_from_gcs(bucket: str, path: str) -> bool:
    try:
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(path)

        if not blob.exists():
            return False

        blob.download_to_filename(settings.CACHE_LOCAL_PATH)
        logger.info("Cache downloaded from GCS")
        return True
    except Exception as e:
        logger.warning(f"Error downloading cache from GCS: {e}")
        return False