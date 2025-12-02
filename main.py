from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from google.cloud import storage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import vertexai
#from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.language_models import TextEmbeddingModel  # ← CAMBIO AQUÍ
from vertexai.generative_models import GenerativeModel
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
FILE_NAME = os.environ.get("FILE_NAME", "gemini-ai/bd_conocimiento.xlsx")
EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "models/gemini-1.5-pro" ## "gemini-1.5-pro" ## 

logger.info(f"Initializing Vertex AI with project: {PROJECT_ID}")
vertexai.init(project=PROJECT_ID, location=LOCATION)
embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
gen_model = GenerativeModel(GEN_MODEL)
logger.info("Vertex AI initialized successfully")

app = FastAPI()

# ====== VAR GLOBALES ======
chunks = None
chunk_embeddings = None

def load_excel_from_gcs(bucket, path):
    logger.info(f"Loading Excel from gs://{bucket}/{path}")
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(path)
    data = blob.download_as_bytes()
    logger.info("Excel loaded successfully")
    return pd.read_excel(data)

def chunk_text(text, max_chars=600):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def embed(text):
    emb = embed_model.get_embeddings([text])
    return np.array(emb[0].values)

# ================================
# ENDPOINTS
# ================================
@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    global chunks, chunk_embeddings
    
    try:
        # ============ 1. Cargar SOLO la 1ra vez ============
        if chunks is None or chunk_embeddings is None:
            logger.info("Loading knowledge base for the first time...")
            df = load_excel_from_gcs(BUCKET, FILE_NAME)
            full_text = "\n".join([str(x) for x in df.values.flatten() if pd.notnull(x)])
            chunks = chunk_text(full_text)
            logger.info(f"Created {len(chunks)} chunks")
            chunk_embeddings = np.vstack([embed(c) for c in chunks])
            logger.info("Embeddings created successfully")
        
        # ============ 2. RAG SEARCH ============
        query_emb = embed(q.question).reshape(1, -1)
        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        top_idx = scores.argsort()[-3:][::-1]
        retrieved_chunks = [chunks[i] for i in top_idx]
        context = "\n\n".join(retrieved_chunks)
        
        prompt = f"""
Usa SOLO el siguiente contexto para responder.
=== CONTEXTO ===
{context}
=== PREGUNTA ===
{q.question}
=== RESPUESTA ===
"""
        llm_response = gen_model.generate_content(prompt)
        return {
            "answer": llm_response.text,
            "chunks_used": retrieved_chunks
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        #return {"error": str(e)}, 500
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete")
    logger.info(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}")
