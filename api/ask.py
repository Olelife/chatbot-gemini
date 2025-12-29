import logging
import random
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from slack_sdk import WebClient
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings
from models.question import Question
from rag.embeddings import embed_single
from rag.generator import generate_answer
from rag.knowledge import build_or_load_knowledge_base
from rag.session_memory import build_history_text, add_to_history
from services.logging_service import save_log_async

import numpy as np
from utils.timer import Timer

slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)
router = APIRouter(prefix="/ask", tags=["Ask"])
logger = logging.getLogger(__name__)

kb_chunks = {}
kb_embeddings = {}
kb_metadata = {}

def should_introduce(history_text: str, question: str) -> bool:
    question_lower = question.lower().strip()
    is_first_message = not history_text.strip()

    is_question = "?" in question_lower
    word_count = len(question_lower.split())

    greetings = ["hola", "oi", "olá", "ola", "buenas", "bom dia", "boa tarde"]
    only_greeting = word_count <= 3 and any(g in question_lower for g in greetings)

    ask_identity = any(term in question_lower for term in [
        "quién eres", "quien eres", "como te llamas",
        "qual é o seu nome", "quem é você", "quem é voce"
    ])

    if is_first_message:
        return True
    if ask_identity:
        return True
    if only_greeting and not is_question:
        return True
    return False

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
    question_lower = question.lower().strip()

    language = "español de México" if country == "mx" else "portugués de Brasil"

    introductions = {
        "mx": [
            "¡Hola! Soy **Olé Assistant**, tu especialista en seguros de vida.",
            "¿Qué tal? Estoy aquí para ayudarte con todo lo relacionado a seguros Olé.",
            "Encantado de apoyarte. Soy Olé Assistant 😄",
        ],
        "br": [
            "Olá! Sou **Olé Assistant**, seu especialista em seguros de vida.",
            "Oi! Estou aqui para te ajudar com tudo da Olé 😄",
            "Prazer em te ajudar, sou Olé Assistant!",
        ]
    }

    intro = ""
    if should_introduce(history_text, question):
        intro = random.choice(introductions[country]) + "\n\n"

    prompt = f"""
    {intro} 
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
    - Responde en idioma: **{language}**
    - Si faltan datos en el contexto, dilo claramente.
    - Mantén coherencia sin repetir información innecesaria.
    - Núcleo de la respuesta basada en el contexto anterior.
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

    if background_tasks is not None:
        background_tasks.add_task(
            save_log_async,
            question,
            answer,
            username,
            retrieved_chunks,
            scores_list,
            {
                "country": country,
                "ip": request.client.host if request else None,
                "user-agent": request.headers.get("User-Agent") if request else None,
            },
        )
    else:
        # Fallback: log de manera síncrona en caso de Slack/evento
        save_log_async(
            question,
            answer,
            username,
            retrieved_chunks,
            scores_list,
            {
                "country": country,
                "ip": request.client.host if request else None,
                "user-agent": request.headers.get("User-Agent") if request else None,
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

async def background_slack_task(
    question: str,
    session_id: str,
    country: str,
    username: str,
    request: Request,
    channel_id: str
):
    try:
        # Procesar la IA
        result = process_question(
            question,
            session_id,
            country,
            username,
            request,
            BackgroundTasks()  # dummy
        )

        answer = result["answer"]

        # Enviar mensaje final a Slack
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"<@{username}> {answer}"
        )

    except Exception as e:
        logger.error(f"Slack processing error: {e}")
        slack_client.chat_postMessage(
            channel=channel_id,
            text="⚠️ Tive um problema processando sua mensagem. Tente novamente!"
        )

# ============================================================
# SLACK API
# ============================================================
@router.post("/slack")
async def ask_slack(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()

    user_id = form.get("user_id")
    question = form.get("text")
    country = form.get("country", "br").lower()
    channel_id = form.get("channel_id")
    session_id = f"slack-{user_id}"

    logger.info(
        msg=f"Asking question {question} for country {country} with user_id {user_id} and session_id {session_id} in channel {channel_id}"
    )

    background_tasks.add_task(
        background_slack_task,
        question,
        session_id,
        country,
        user_id,
        request,
        channel_id
    )

    return PlainTextResponse("⏳ Estou analisando sua pergunta...")
