# app/rag/search.py
from typing import List, Dict, Any
from rag.faiss_index import search_faiss

def search_top_k(query_vector, vectors, chunks, k=3):
    scores, idxs = search_faiss(query_vector, k)

    results = []
    for score, idx in zip(scores, idxs):
        results.append({
            "chunk_index": int(idx),
            "score": float(score),
            "chunk": chunks[idx]
        })
    return results
