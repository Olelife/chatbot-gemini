from fastapi import FastAPI, Request
from google.cloud import storage
import pandas as pd
from vertexai.generative_models import GenerativeModel
import vertexai
import os

app = FastAPI()

PROJECT_ID = os.environ.get("PROJECT_ID", "olelifetech")
BUCKET = os.environ.get("BUCKET", "olelife-lakehouse")

# PATH CORRECTO: carpeta + archivo
FILE_NAME = os.environ.get("FILE_NAME", "gemini-ai/bd_conocimiento.xlsx")

vertexai.init(project=PROJECT_ID, location="us-central1")
model = GenerativeModel("gemini-1.5-pro")

def load_kb():
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET)
        blob = bucket.blob(FILE_NAME)

        # Cloud Run SOLO permite /tmp
        local_file = "/tmp/bd_conocimiento.xlsx"
        blob.download_to_filename(local_file)

        df = pd.read_excel(local_file)

        if "titulo" not in df.columns or "contenido" not in df.columns:
            raise Exception("El Excel debe tener columnas: titulo, contenido")

        return df

    except Exception as e:
        print("ERROR CARGANDO EXCEL:", e)
        return pd.DataFrame(columns=["titulo", "contenido"])

KB = load_kb()

def search_kb(question: str):
    if KB.empty:
        return "No existe información en la base de conocimiento."

    question_low = question.lower()
    results = []

    for _, row in KB.iterrows():
        score = question_low.count(str(row["titulo"]).lower())
        results.append((score, row["contenido"]))

    results = sorted(results, reverse=True)
    return results[0][1] if results else "No encontré información relacionada."

@app.post("/ask")
async def ask(req: Request):
    data = await req.json()
    question = data["question"]

    context = search_kb(question)

    prompt = f"""
    Usa exclusivamente esta base de conocimiento para responder.
    CONTEXTO:
    {context}

    Pregunta:
    {question}

    Respuesta concreta basada SOLO en el contexto:
    """

    response = model.generate_content(prompt)
    return {"answer": response.text}
