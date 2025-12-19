import os
import time
import logging
import pickle
import hashlib
from google.cloud import storage

logger = logging.getLogger(__name__)


def save_cache_local(chunks_list, embeddings, path: str):
    try:
        metadata = {
            "created_at": time.time(),
            "chunks_count": len(chunks_list),
            "embedding_dim": embeddings.shape[1] if embeddings.size else 0,
            "hash": hashlib.sha256("\n".join(chunks_list).encode()).hexdigest(),
        }

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": chunks_list,
                    "embeddings": embeddings,
                    "metadata": metadata,
                },
                f,
            )

        logger.info(f"Local cache saved → {path}")

    except Exception as e:
        logger.warning(f"Could not save local cache: {e}")


def load_cache_local(path: str):
    if not os.path.exists(path):
        return None, None, None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        logger.info(f"Local cache loaded → {path}")
        return data.get("chunks"), data.get("embeddings"), data.get("metadata")

    except Exception as e:
        logger.warning(f"Could not load local cache: {e}")
        return None, None, None


def upload_cache_to_gcs(bucket: str, gcs_path: str, local_path: str):
    try:
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"GCS cache uploaded → gs://{bucket}/{gcs_path}")
    except Exception as e:
        logger.warning(f"Could not upload GCS cache: {e}")


def download_cache_from_gcs(bucket: str, gcs_path: str, local_path: str):
    try:
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(gcs_path)

        if not blob.exists():
            return False

        blob.download_to_filename(local_path)
        return True

    except Exception as e:
        logger.warning(f"Error downloading cache from GCS: {e}")
        return False
