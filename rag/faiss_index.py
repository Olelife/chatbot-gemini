# app/rag/faiss_index.py
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

index = None
dimension = None
faiss_ready = False


def build_faiss_index(embeddings: np.ndarray, use_hnsw=True):
    """
    Construye un índice FAISS para búsquedas ultra rápidas.
    Usa HNSW si use_hnsw=True, caso contrario usa IndexFlatIP (exacto).
    """

    global index, dimension, faiss_ready

    dimension = embeddings.shape[1]

    if use_hnsw:
        # HNSW = Hierarchical Navigable Small World Graph
        # MUY ORDENADO: eficiencia O(log N)
        logger.info("Building HNSW FAISS index...")
        hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # efConstruction=32

        # Para similitud coseno necesitamos vectores normalizados
        faiss.normalize_L2(embeddings)

        hnsw_index.hnsw.efSearch = 64  # mejora calidad
        hnsw_index.add(embeddings)

        index = hnsw_index
        faiss_ready = True
        logger.info("FAISS HNSW index built successfully")

    else:
        # Index exacto (más lento que HNSW para muchos docs)
        logger.info("Building FLAT FAISS index...")
        flat_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        flat_index.add(embeddings)
        index = flat_index
        faiss_ready = True
        logger.info("FAISS FLAT index built successfully")

    return index


def search_faiss(query_vector: np.ndarray, k: int = 5):
    """
    Realiza búsqueda FAISS top-k.
    """
    global index, faiss_ready

    if not faiss_ready or index is None:
        raise RuntimeError("FAISS index not initialized")

    # Normalizamos también la query
    vec = query_vector.astype("float32").reshape(1, -1)
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, k)

    return scores[0], indices[0]
