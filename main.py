from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
from google.cloud import storage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import vertexai
from vertexai.building_blocks import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# ================================
# CONFIG UP
# ================================
PROJECT_ID = os.environ.get("PROJECT_ID", "olelifetech")
LOCATION = "us-central1"
BUCKET = os.environ.get("BUCKET", "olelife-lakehouse")
FILE_NAME = os.environ.get("FILE_NAME", "gemini-ai/bd_conocimiento.xlsx")

EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-1.5-pro"

vertexai.init(project=PROJECT_ID, location=LOCATION)

embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
gen_model = GenerativeModel(GEN_MODEL)

app = FastAPI()

# ====== VAR GLOBALES ======
chunks = None
chunk_embeddings = None

def load_excel_from_gcs(bucket, path):
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(path)
    data = blob.download_as_bytes()
    return pd.read_excel(data)

def chunk_text(text, max_chars=600):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def embed(text):
    emb = embed_model.get_embeddings([text])
    return np.array(emb[0].values)

# ================================
# ENDPOINT
# ================================
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    global chunks, chunk_embeddings

    # ============ 1. Cargar SOLO la 1ra vez ============
    if chunks is None or chunk_embeddings is None:
        df = load_excel_from_gcs(BUCKET, FILE_NAME)
        full_text = "\n".join([str(x) for x in df.values.flatten() if pd.notnull(x)])
        chunks = chunk_text(full_text)
        chunk_embeddings = np.vstack([embed(c) for c in chunks])

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
