from fastapi import APIRouter, Request, HTTPException
from models.question import Question
from services.rag_service import process_question
from services.logging_service import save_log
import time

router = APIRouter(prefix="/ask", tags=["Ask"])


@router.post("")
def ask(q: Question, request: Request):
    timings = {}  # ← almacenará tiempos por etapa
    t0 = time.time()

    try:
        # 1. Procesar la pregunta (embedding + faiss + generación)
        t1 = time.time()
        answer, chunks_used, scores, rag_timings = process_question(q.question)
        t2 = time.time()

        timings["processing"] = t2 - t1

        # 2. Guardar logs en Postgres
        t3 = time.time()
        save_log(
            question=q.question,
            answer=answer,
            username=request.state.username,
            chunks_used=chunks_used,
            scores=scores,
            metadata={
                "ip": request.client.host,
                "user-agent": request.headers.get("user-agent")
            }
        )
        t4 = time.time()

        timings["db_log"] = t4 - t3
        timings["total"] = t4 - t0
        timings["rag"] = rag_timings

        return {
            "answer": answer,
            "chunks_used": chunks_used,
            "scores": scores,
            "timings": timings   # ← ahora lo devolvemos
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
