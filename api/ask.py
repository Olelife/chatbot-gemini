import logging
from typing import Optional, List

from fastapi import APIRouter, Request, HTTPException
from sklearn.metrics.pairwise import cosine_similarity

from models.question import Question
from rag.generator import generate_answer
from rag.knowledge import build_or_load_knowledge_base, embed_single
from rag.session_memory import build_history_text, add_to_history
from services.logging_service import save_log

import numpy as np

router = APIRouter(prefix="/ask", tags=["Ask"])
logger = logging.getLogger(__name__)
chunks: Optional[List[str]] = None
chunk_embeddings: Optional[np.ndarray] = None

@router.post("")
def ask(q: Question, request: Request):
    username = request.headers.get("x-username")
    session_id = request.headers.get("x-session-id", "default")
    global chunks, chunk_embeddings, kb_metadata

    try:
        if chunks is None:
            chunks, chunk_embeddings, kb_metadata = build_or_load_knowledge_base()

        query_emb = embed_single(q.question).reshape(1, -1)

        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        top_idx = scores.argsort()[-3:][::-1]
        retrieved_chunks = [chunks[i] for i in top_idx]

        context = "\n\n".join(retrieved_chunks)

        history_text = build_history_text(session_id)

        prompt = f"""
Soy **Olé Assistant**, tu asistente profesional especializado en seguros de vida,
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

============= HISTORIAL DE LA CONVERSACIÓN =============
{history_text}

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

        # ==== LOG EN BASE DE DATOS ====
        scores_list = [
            {"chunk_index": int(i), "score": float(scores[i])}
            for i in top_idx
        ]

        save_log(
            question=q.question,
            answer=answer,
            username=username,
            chunks_used=retrieved_chunks,
            scores=scores_list,
            metadata={
                "ip": request.client.host,
                "user-agent": request.headers.get("User-Agent"),
            }
        )

        add_to_history(session_id, "user", q.question)
        add_to_history(session_id, "assistant", answer)

        return {
            "answer": answer,
            "chunks_used": retrieved_chunks,
            "scores": scores_list
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))