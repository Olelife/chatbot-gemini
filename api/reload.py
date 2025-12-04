from fastapi import APIRouter
from rag.knowledge import rebuild_knowledge

router = APIRouter(prefix="/reload", tags=["Knowledge"])

@router.post("")
def reload_kb():
    chunks, vectors, meta = rebuild_knowledge()
    return {
        "status": "ok",
        "chunks": len(chunks),
        "embedding_dim": vectors.shape[1],
        "metadata": meta
    }
