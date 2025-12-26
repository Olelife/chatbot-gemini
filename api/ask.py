import logging
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from sklearn.metrics.pairwise import cosine_similarity

from models.question import Question
from rag.embeddings import embed_single
from rag.generator import generate_answer
from rag.knowledge import build_or_load_knowledge_base
from rag.session_memory import build_history_text, add_to_history
from services.logging_service import save_log_async

import numpy as np
from utils.timer import Timer

router = APIRouter(prefix="/ask", tags=["Ask"])
logger = logging.getLogger(__name__)

kb_chunks = {}
kb_embeddings = {}
kb_metadata = {}


def process_question(
    question: str,
    session_id: str,
    country: str,
    username: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    timer = Timer()
    timer.start("start")

    timer.start("load_kb")
    if country not in kb_chunks:
        chunks, embeddings, metadata = build_or_load_knowledge_base(country)
        kb_chunks[country] = chunks
        kb_embeddings[country] = embeddings
        kb_metadata[country] = metadata
    timer.end("load_kb")

    chunks = kb_chunks[country]
    chunk_embeddings = kb_embeddings[country]

    timer.start("embedding")
    query_emb = embed_single(question).reshape(1, -1)
    timer.end("embedding")

    timer.start("vector_search")
    scores = cosine_similarity(query_emb, chunk_embeddings)[0]
    top_idx = scores.argsort()[-3:][::-1]
    retrieved_chunks = [chunks[i] for i in top_idx]
    timer.end("vector_search")

    timer.start("prompt_build")
    context = "\n\n".join(retrieved_chunks)
    history_text = build_history_text(session_id, country)

    language = "español de México" if country == "mx" else "portugués de Brasil"

    prompt = f"""
Eres **Olé Assistant**, un asistente experto en seguros de vida. 
Debes mantener coherencia a lo largo de la conversación y responder 
solo con información contenida en el contexto RAG.

==================================================
🧠 MEMORIA DE LA CONVERSACIÓN
==================================================
{history_text}

==================================================
📘 CONTEXTO OFICIAL (RAG)
==================================================
{context}

==================================================
🎯 INSTRUCCIONES
==================================================
- Mantén coherencia, sin repetirte.
- Responde en idioma: {language}
- Si no hay contexto, dilo.
- No inventes información.

==================================================
❓ PREGUNTA DEL USUARIO
==================================================
{question}

==================================================
📝 RESPUESTA
==================================================
"""
    timer.end("prompt_build")

    timer.start("generation")
    answer = generate_answer(prompt)
    timer.end("generation")

    # Logging DB async
    timer.start("db_log")
    scores_list = [
        {"chunk_index": int(i), "score": float(scores[i])}
        for i in top_idx
    ]

    background_tasks.add_task(
        save_log_async,
        question,
        answer,
        username,
        retrieved_chunks,
        scores_list,
        {
            "country": country,
            "ip": request.client.host,
            "user-agent": request.headers.get("User-Agent"),
        },
    )
    timer.end("db_log")

    # Guardar en memoria conversacional
    add_to_history(session_id, country, "user", question)
    add_to_history(session_id, country, "assistant", answer)

    return {
        "answer": answer,
        "chunks_used": retrieved_chunks,
        "scores": scores_list,
        "timings": timer.summary(),
        "country": country,
    }


# ============================================================
# API NORMAL (Web App)
# ============================================================
@router.post("")
def ask(q: Question, request: Request, background_tasks: BackgroundTasks):
    username = request.headers.get("x-username")
    session_id = request.headers.get("x-session-id", "default")
    country = request.headers.get("x-country", "mx").lower()

    return process_question(
        q.question,
        session_id,
        country,
        username,
        request,
        background_tasks,
    )


# ============================================================
# SLACK API
# ============================================================
@router.post("/slack")
async def ask_slack(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()

    user_id = form.get("user_id")
    channel_id = form.get("channel_id")
    question = form.get("text")
    # username = form.get("user_id", "slack-user")
    # session_id = form.get("channel_id", "slack-session")
    country = form.get("country", "br").lower()

    session_id = f"slack-{user_id}"

    result = process_question(
        question,
        session_id,
        country,
        user_id,
        request,
        background_tasks,
    )

    return {"text": result["answer"]}
