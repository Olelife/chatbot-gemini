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

@router.post("")
def ask(q: Question, request: Request, background_tasks: BackgroundTasks):
    timer = Timer()
    timer.start("start")

    username = request.headers.get("x-username")
    session_id = request.headers.get("x-session-id", "default")
    country = request.headers.get("x-country", "mx").lower()

    try:
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
        query_emb = embed_single(q.question).reshape(1, -1)
        timer.end("embedding")

        timer.start("vector_search")
        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        top_idx = scores.argsort()[-3:][::-1]
        retrieved_chunks = [chunks[i] for i in top_idx]
        timer.end("vector_search")

        timer.start("prompt_build")
        context = "\n\n".join(retrieved_chunks)
        history_text = build_history_text(session_id, country)

        language = 'español de méxico'
        if country == 'br' :
            language = 'portugués de Brasil'

        prompt = f"""
        Eres **Olé Assistant**, un asistente experto en seguros de vida. 
        Debes mantener coherencia a lo largo de la conversación y responder 
        solo con información contenida en el contexto RAG.

        ==================================================
        🧠 MEMORIA DE LA CONVERSACIÓN
        (Usa este historial para mantener continuidad, tono y coherencia)
        ==================================================
        {history_text}

        ==================================================
        📘 CONTEXTO OFICIAL (RAG)
        (Usa estos datos como única fuente de verdad. No inventes nada.)
        ==================================================
        {context}

        ==================================================
        🎯 INSTRUCCIONES PARA GENERAR LA RESPUESTA
        ==================================================
        - Si la pregunta del usuario se relaciona con mensajes previos, continúa el hilo.
        - Si la pregunta inicia un nuevo tema, respóndelo sin referirse a la memoria.
        - No repitas información ya dada en turnos anteriores.
        - Mantén un estilo profesional, cálido y directo.
        - No inventes datos fuera del contexto RAG.
        - Si faltan datos, dilo de manera clara y profesional.
        - Si existen reglas diferentes por cobertura o sección, sepáralas claramente.
        - La respuesta tiene que estar en idioma {language}

        ==================================================
        ❓ PREGUNTA ACTUAL DEL USUARIO
        ==================================================
        {q.question}

        ==================================================
        📝 GENERA LA RESPUESTA
        ==================================================
        """
        timer.end("prompt_build")

        timer.start("generation")
        print(prompt)
        answer = generate_answer(prompt)
        timer.end("generation")

        # ==== LOG EN BASE DE DATOS ====
        timer.start("db_log")
        scores_list = [
            {"chunk_index": int(i), "score": float(scores[i])}
            for i in top_idx
        ]

        background_tasks.add_task(
            save_log_async,
            q.question,
            answer,
            username,
            retrieved_chunks,
            scores_list,
            {
                "country": country,
                "ip": request.client.host,
                "user-agent": request.headers.get("User-Agent"),
            }
        )
        timer.end("db_log")

        timings = timer.summary()

        add_to_history(session_id, country, "user", q.question)
        add_to_history(session_id, country, "assistant", answer)

        return {
            "answer": answer,
            "chunks_used": retrieved_chunks,
            "scores": scores_list,
            "timings": timings,
            "country": country,
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))