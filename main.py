import os
import io
import logging
import pickle
from contextlib import asynccontextmanager
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import storage

from google import genai
from google.genai import types  # EmbedContentConfig, etc.

import time
import hashlib

# ================================
# LOGGING
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CONFIG
# ================================
PROJECT_ID = os.environ.get("PROJECT_ID", "olelifetech")
LOCATION = os.environ.get("LOCATION", "us-central1")

BUCKET = os.environ.get("BUCKET", "olelife-lakehouse")
KNOWLEDGE_FOLDER = "gemini-ai/knowledge"   # NUEVO

EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-2.5-pro")

CACHE_LOCAL_PATH = "/tmp/embedding_cache.pkl"
CACHE_GCS_PATH = "gemini-ai/embedding_cache.pkl"

# ================================
# VARIABLES GLOBALES
# ================================
genai_client: Optional[genai.Client] = None
chunks: Optional[List[str]] = None
chunk_embeddings: Optional[np.ndarray] = None


# ================================
# UTILS GENERALES
# ================================
def list_files_in_folder(bucket: str, folder: str):
    """Lista todos los archivos dentro de una carpeta en GCS."""
    client = storage.Client()
    bucket_obj = client.bucket(bucket)

    return [
        blob.name
        for blob in bucket_obj.list_blobs(prefix=folder)
        if not blob.name.endswith("/")
    ]


def load_file_as_text(bucket: str, path: str) -> str:
    """Carga un archivo JSON o TXT desde GCS y lo retorna como texto plano."""
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(path)

    if not blob.exists():
        logger.warning(f"File not found: {path}")
        return ""

    raw = blob.download_as_bytes()

    if path.endswith(".json"):
        try:
            import json
            data = json.loads(raw.decode("utf-8"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error reading JSON {path}: {e}")
            return ""

    try:
        return raw.decode("utf-8")
    except:
        return str(raw)

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

# ================================
# CACHE
# ================================
def save_cache_local(chunks_list, embeddings):
    try:
        metadata = {
            "created_at": time.time(),
            "chunks_count": len(chunks_list),
            "embedding_dim": embeddings.shape[1] if embeddings.size else 0,
            "hash": hashlib.sha256("\n".join(chunks_list).encode("utf-8")).hexdigest(),
        }

        with open(CACHE_LOCAL_PATH, "wb") as f:
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
    if not os.path.exists(CACHE_LOCAL_PATH):
        return None, None, None

    try:
        with open(CACHE_LOCAL_PATH, "rb") as f:
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
        blob.upload_from_filename(CACHE_LOCAL_PATH)
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

        blob.download_to_filename(CACHE_LOCAL_PATH)
        logger.info("Cache downloaded from GCS")
        return True
    except Exception as e:
        logger.warning(f"Error downloading cache from GCS: {e}")
        return False


# ================================
# GENAI CLIENT + EMBEDDINGS
# ================================
def ensure_genai_client():
    global genai_client
    if genai_client is None:
        logger.info("Initializing GenAI Vertex client...")
        genai_client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
    return genai_client


def embed_texts(texts: List[str]) -> np.ndarray:
    client = ensure_genai_client()

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    vectors = [np.array(e.values, dtype="float32") for e in response.embeddings]
    return np.vstack(vectors)


def embed_single(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def generate_answer(prompt: str) -> str:
    client = ensure_genai_client()
    resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return resp.text


def chunk_text(text: str, max_chars=600):
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


# ================================
# KNOWLEDGE BUILDER
# ================================
def build_or_load_knowledge_base():
    # 1) Intentar cache local
    local_chunks, local_emb, metadata = load_cache_local()
    if local_chunks is not None:
        logger.info("Loaded KB from LOCAL CACHE")
        return local_chunks, local_emb, metadata

    # 2) Intentar cache GCS
    if download_cache_from_gcs(BUCKET, CACHE_GCS_PATH):
        c, e, m = load_cache_local()
        if c is not None:
            logger.info("Loaded KB from GCS CACHE")
            return c, e, m

    # 3) Construir desde knowledge/
    logger.info("Building KB from raw JSON knowledge folder…")

    kb_chunks = build_knowledge_units_from_folder(BUCKET, KNOWLEDGE_FOLDER)
    logger.info(f"KB semantic chunks: {len(kb_chunks)}")
    kb_embeddings = embed_texts(kb_chunks)

    save_cache_local(kb_chunks, kb_embeddings)
    upload_cache_to_gcs(BUCKET, CACHE_GCS_PATH)

    # cargar metadata recien guardada
    _, _, metadata = load_cache_local()

    return kb_chunks, kb_embeddings, metadata

# ================================
# FASTAPI
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Startup")
    ensure_genai_client()
    yield
    logger.info("API Shutdown")


app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    global chunks, chunk_embeddings, kb_metadata

    try:
        if chunks is None:
            chunks, chunk_embeddings, kb_metadata = build_or_load_knowledge_base()

        query_emb = embed_single(q.question).reshape(1, -1)

        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        top_idx = scores.argsort()[-3:][::-1]
        retrieved_chunks = [chunks[i] for i in top_idx]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
Soy **Coach OleLife**, tu asistente profesional especializado en seguros de vida,
procesos operativos y uso de la plataforma OleLife. Estoy aquí para ayudarte de
forma clara, confiable y precisa.

Mi estilo es:
- corporativo y profesional (como un asesor experto de OleLife)
- cálido y humano sin usar frases genéricas ni repetitivas
- en primera persona (“te explico”, “puedo ayudarte”, “esto aplica en tu caso”)
- flexible, directo y contextual según la conversación

==================================================
REGLAS DE COMPORTAMIENTO
==================================================
1. Respondo **solo** con lo que esté en el contexto recuperado; no invento datos.
2. Si existe más de una edad, requisito o regla:
   - explico cada una por cobertura o sección correspondiente
   - nunca mezclo valores de coberturas diferentes
3. Si la información no está en el contexto:
   - digo: “Según la información disponible, no tengo una respuesta exacta para eso…”
4. Si el usuario me pide guiarlo (ej. cotizar o seguir un proceso):
   - formulo preguntas en orden lógico
   - pido un dato por vez
   - mantengo claridad y precisión en las instrucciones
5. **Memoria conversacional ligera:**
   - detecto si el usuario está siguiendo un hilo del tema
   - evito repetir información que ya mencioné en esta conversación
   - no uso saludos en turnos posteriores
   - adapto el detalle según lo ya conversado
   - si el usuario pide una aclaración, solo amplío lo necesario
6. Variación conversacional:
   - no empiezo siempre igual
   - puedo usar distintas formas de introducir una respuesta:
     “Sobre ese punto…”, “Esto es lo que aplica…”, “Según el contexto…”
7. Tono corporativo humano:
   - comunico claridad, profesionalismo y confianza
   - uso listas solo cuando ayudan a la comprensión
   - evito tecnicismos innecesarios, explico en lenguaje simple

==================================================
CONTEXTO CERTIFICADO (RAG)
==================================================
Este contenido fue recuperado desde la base de conocimiento oficial.
Toda la respuesta debe basarse estrictamente en esto.

{context}

===============================
PREGUNTA DEL USUARIO
===============================
{q.question}

===============================
RESPUESTA
===============================
"""

        answer = generate_answer(prompt)

        return {
            "answer": answer,
            "chunks_used": retrieved_chunks,
            "scores": [float(scores[i]) for i in top_idx],
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-kb")
def debug_kb():
    global chunks, chunk_embeddings, kb_metadata

    if chunks is None:
        return {"error": "KB not loaded yet"}

    return {
        "status": "ok",
        "chunks_count": len(chunks),
        "embedding_dim": int(chunk_embeddings.shape[1]),
        "metadata": kb_metadata,
        "sample_chunks": chunks[:3],   # primeros 3 chunks para validar
    }

@app.get("/debug-files")
def debug_files():
    """
    Lista todos los archivos dentro de gemini-ai/knowledge que la API está usando
    para construir la base de conocimiento.
    """
    try:
        files = list_files_in_folder(BUCKET, KNOWLEDGE_FOLDER)

        # Filtrar carpetas como prompts automáticamente
        filtered = [f for f in files if "prompts" not in f]

        return {
            "status": "ok",
            "folder": KNOWLEDGE_FOLDER,
            "total_files": len(filtered),
            "files": filtered,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-search")
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

@app.post("/reload-kb")
def reload_kb():
    """
    Fuerza la recarga completa de la base de conocimiento:
    - Borra cache local
    - Borra cache en GCS
    - Reconstruye chunks + embeddings
    """
    global chunks, chunk_embeddings, kb_metadata

    # 1. Borrar cache local
    if os.path.exists(CACHE_LOCAL_PATH):
        os.remove(CACHE_LOCAL_PATH)

    # 2. Borrar cache en GCS
    try:
        client = storage.Client()
        bucket_obj = client.bucket(BUCKET)
        blob = bucket_obj.blob(CACHE_GCS_PATH)
        if blob.exists():
            blob.delete()
    except Exception as e:
        return {"error": f"Could not delete GCS cache: {e}"}

    # 3. Reconstruir KB desde cero
    try:
        chunks, chunk_embeddings, kb_metadata = build_or_load_knowledge_base()
    except Exception as e:
        return {"error": f"Failed to rebuild KB: {e}"}

    return {
        "status": "reloaded",
        "chunks_count": len(chunks),
        "embedding_dim": int(chunk_embeddings.shape[1]),
        "metadata": kb_metadata,
    }

@app.get("/debug-config")
def debug_config():
    """
    Muestra la configuración general del sistema:
    - Variables de entorno
    - Modelos
    - Ubicaciones de rutas
    - Configuración del bucket GCS
    """
    config = {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "bucket": BUCKET,
        "knowledge_folder": KNOWLEDGE_FOLDER,
        "embedding_model": EMBED_MODEL,
        "generative_model": GEN_MODEL,
        "cache_local_path": CACHE_LOCAL_PATH,
        "cache_gcs_path": CACHE_GCS_PATH,
    }

    # incluir info de la KB si ya está cargada
    if chunks is not None:
        config["chunks_loaded"] = len(chunks)
    if chunk_embeddings is not None:
        config["embedding_dim"] = int(chunk_embeddings.shape[1])

    return config

@app.get("/explain-search")
def explain_search(q: str):
    """
    Explica paso por paso cómo funciona la búsqueda vectorial.
    No llama al modelo generativo.
    """
    global chunks, chunk_embeddings, kb_metadata

    if chunks is None:
        chunks, chunk_embeddings, kb_metadata = build_or_load_knowledge_base()

    try:
        # 1) Embedding de la query
        query_embedding = embed_single(q)

        # 2) Similaridad coseno contra todos los chunks
        scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk_embeddings
        )[0]

        # 3) Top 5 chunks relevantes
        top_idx = scores.argsort()[-5:][::-1]

        detailed = []
        for idx in top_idx:
            detailed.append({
                "chunk_index": int(idx),
                "score": float(scores[idx]),
                "text_preview": chunks[idx][:300],
                "reason": "Alta similitud coseno respecto a la consulta."
            })

        return {
            "query": q,
            "query_embedding_preview": list(query_embedding[:10]), # primeros 10 valores
            "total_chunks": len(chunks),
            "top_matches": detailed,
            "max_score": float(scores[top_idx[0]]),
            "min_score": float(scores.min()),
            "avg_score": float(scores.mean()),
            "notes": [
                "Este endpoint no usa el modelo generativo.",
                "Solo evalúa embeddings + similitud coseno.",
                "Útil para tuning de RAG."
            ]
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
