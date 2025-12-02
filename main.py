from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from google.cloud import storage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import vertexai
from vertexai.language_models import TextEmbeddingModel  
from vertexai.generative_models import GenerativeModel
import google.cloud.aiplatform
import logging

# Configurar loggingßßß
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CONFIG UPP
# ================================
PROJECT_ID = os.environ.get("PROJECT_ID", "olelifetech")
LOCATION = "us-central1"
BUCKET = os.environ.get("BUCKET", "olelife-lakehouse")
PREFIX = os.environ.get("PREFIX", "gemini-ai/knowledge/mx/") ## NEW
FILE_NAME = os.environ.get("FILE_NAME", "gemini-ai/bd_conocimiento.xlsx")
EMBED_MODEL = "text-embedding-005"
GEN_MODEL = "models/gemini-2.5-pro" 

# Archivo maestro
INDEX_FILE = "knowledge/mx/index.json"

logger.info(f"Initializing Vertex AI with project: {PROJECT_ID}")
vertexai.init(project=PROJECT_ID, location=LOCATION)
embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
gen_model = GenerativeModel(GEN_MODEL)
logger.info("Vertex AI initialized successfully")

app = FastAPI()

# ====== VAR GLOBALES ======
#chunks = None
#chunk_embeddings = None

#chunks = []               # lista de textos cortados
#chunk_sources = []        # path de cada chunk
#chunk_embeddings = None   # embeddings np.array

# Variables globales
KB_TEXTS = []           # Lista de textos
KB_SOURCES = []         # Lista de metadatos {"module":..., "file":...}
EMBEDDINGS = None       # Matriz de embeddings


# ==========================================================
# UTILIDADES
# ==========================================================
def gcs_read_text(path: str) -> str:
    """Lee un archivo de GCS como texto."""
    client = storage.Client()
    blob = client.bucket(BUCKET).blob(path)
    return blob.download_as_text()


def load_index() -> dict:
    """Carga el index.json desde GCS."""
    logger.info(f"📥 Cargando index.json desde gs://{BUCKET}/{INDEX_FILE}")
    index_data = gcs_read_text(INDEX_FILE)
    return json.loads(index_data)


def flatten_modules(modules_dict):
    """
    Convierte el árbol de modules del index.json en una lista plana:
    [
        {"module": "faq", "file": "knowledge/mx/faq.json"},
        {"module": "coberturas.cancer", "file": "knowledge/mx/coberturas/cancer.json"},
        ...
    ]
    """
    flattened = []

    def recurse(prefix, obj):
        if isinstance(obj, str):
            flattened.append({"module": prefix, "file": obj})
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                recurse(new_prefix, v)

    recurse("", modules_dict)
    return flattened


def chunk_text(text, max_chars=700):
    """Divide texto largo en chunks."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def embed(text: str):
    """Devuelve embedding para un texto."""
    vec = embed_model.get_embeddings([text])[0].values
    return np.array(vec)


# ==========================================================
# CARGA DE LA BASE DE CONOCIMIENTO
# ==========================================================
def build_knowledge_base():
    global KB_TEXTS, KB_SOURCES, EMBEDDINGS

    logger.info("🚀 Construyendo Knowledge Base desde index.json...")

    index = load_index()
    modules = flatten_modules(index["modules"])

    all_chunks = []
    all_sources = []

    for mod in modules:
        module_name = mod["module"]
        file_path = mod["file"]

        logger.info(f"📄 Cargando módulo [{module_name}] → {file_path}")

        # Leer archivo
        raw = gcs_read_text(file_path)

        # JSON → convertir en texto
        if file_path.endswith(".json"):
            data = json.loads(raw)
            text = json.dumps(data, ensure_ascii=False, indent=2)

        # TXT → se toma tal cual
        else:
            text = raw

        # Chunking
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            all_sources.append({"module": module_name, "file": file_path})

    # Guardar en memoria
    KB_TEXTS = all_chunks
    KB_SOURCES = all_sources

    # Embeddings
    logger.info("🧠 Generando embeddings...")
    EMBEDDINGS = np.vstack([embed(t) for t in KB_TEXTS])

    logger.info(f"✔ Knowledge Base lista: {len(KB_TEXTS)} chunks cargados")


# ==========================================================
# API MODELOS
# ==========================================================
class Question(BaseModel):
    question: str


@app.on_event("startup")
async def startup():
    logger.info("🔥 Iniciando API…")
    build_knowledge_base()
    logger.info("✔ Backend listo")


@app.get("/")
def health_root():
    return {"status": "ok", "message": "RAG API running", "chunks": len(KB_TEXTS)}


@app.post("/ask")
def ask(q: Question):
    global KB_TEXTS, EMBEDDINGS, KB_SOURCES

    try:
        # Embedding de la pregunta
        q_emb = embed(q.question).reshape(1, -1)

        # Similaridad coseno
        scores = cosine_similarity(q_emb, EMBEDDINGS)[0]
        top_idx = scores.argsort()[-3:][::-1]

        retrieved = [
            {
                "chunk": KB_TEXTS[i],
                "source": KB_SOURCES[i],
                "score": float(scores[i])
            }
            for i in top_idx
        ]

        # Construcción del contexto
        context = "\n\n".join([r["chunk"] for r in retrieved])

        prompt = f"""
Eres un asistente experto. Usa EXCLUSIVAMENTE el siguiente contexto:

{context}

Pregunta:
{q.question}

Respuesta clara y amigable:
"""

        llm_response = gen_model.generate_content(prompt)

        return {
            "answer": llm_response.text,
            "sources": retrieved
        }

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete")
    logger.info(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}")
