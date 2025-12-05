from fastapi import APIRouter
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings
from rag.knowledge import build_or_load_knowledge_base, embed_single
from utils.file import list_files_in_folder

router = APIRouter(prefix="/debug", tags=["Debug"])

@router.get("/config")
def debug_config():
    """
    Muestra la configuración general del sistema:
    - Variables de entorno
    - Modelos
    - Ubicaciones de rutas
    - Configuración del bucket GCS
    """
    config = {
        "project_id": settings.PROJECT_ID,
        "location": settings.LOCATION,
        "bucket": settings.BUCKET,
        "knowledge_folder": settings.KNOWLEDGE_FOLDER,
        "embedding_model": settings.EMBED_MODEL,
        "generative_model": settings.GEN_MODEL,
        "cache_local_path": settings.CACHE_LOCAL_PATH,
        "cache_gcs_path": settings.CACHE_GCS_PATH,
    }

    return config

@router.get("/files")
def debug_files():
    """
    Lista todos los archivos dentro de gemini-ai/knowledge que la API está usando
    para construir la base de conocimiento.
    """
    try:
        files = list_files_in_folder(settings.BUCKET, settings.KNOWLEDGE_FOLDER)

        # Filtrar carpetas como prompts automáticamente
        filtered = [f for f in files if "prompts" not in f]

        return {
            "status": "ok",
            "folder": settings.KNOWLEDGE_FOLDER,
            "total_files": len(filtered),
            "files": filtered,
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/search")
def debug_search(q: str):
    """
    Realiza SOLO la búsqueda vectorial (sin llamar al modelo generativo).
    Esto permite inspeccionar qué chunks del conocimiento se están recuperando.
    """
    global chunks, chunk_embeddings, kb_metadata

    if chunks is None:
        chunks, chunk_embeddings, kb_metadata = build_or_load_knowledge_base()

    try:
        query_emb = embed_single(q).reshape(1, -1)

        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        top_idx = scores.argsort()[-5:][::-1]  # top 5 chunks más relevantes

        results = []
        for idx in top_idx:
            results.append({
                "chunk_index": int(idx),
                "score": float(scores[idx]),
                "text": chunks[idx][:500]  # preview: primeros 500 chars
            })

        return {
            "query": q,
            "top_chunks": results,
        }

    except Exception as e:
        return {"error": str(e)}