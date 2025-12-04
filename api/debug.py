# app/api/debug.py
from fastapi import APIRouter
from rag.knowledge import get_kb, list_files
from rag.embeddings import embed_single
from rag.search import search_top_k

router = APIRouter(prefix="/debug", tags=["Debug"])


@router.get("/kb")
def debug_kb():
    chunks, vectors, meta = get_kb()
    return {
        "chunks": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
        "metadata": meta,
        "sample": chunks[:3],
    }


@router.get("/files")
def debug_files():
    return {"files": list_files()}


@router.get("/search")
def debug_search(q: str):
    chunks, vectors, _ = get_kb()
    vec = embed_single(q)
    matches = search_top_k(vec, vectors, chunks, k=5)
    return {
        "query": q,
        "top_chunks": [
            {
                "chunk_index": m["chunk_index"],
                "score": m["score"],
                "text": m["chunk"][:500],
            }
            for m in matches
        ],
    }
