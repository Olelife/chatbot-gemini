# services/rag_service.py

import time
from typing import List, Dict, Any, Tuple

from rag.embeddings import embed_single
from rag.search import search_top_k
from rag.generator import generate_answer
from rag.knowledge import get_kb
from rag.prompt import build_prompt


def process_question(question: str) -> Tuple[str, List[str], List[Dict[str, float]], Dict[str, float]]:
    """
    Orquesta todo el flujo RAG:
    - carga/coge la KB (chunks + embeddings)
    - calcula embedding de la pregunta
    - hace búsqueda vectorial (FAISS)
    - construye el prompt
    - llama al modelo generativo (Flash/Pro)
    - devuelve respuesta, chunks usados, scores y timings
    """

    timings: Dict[str, float] = {}
    t0 = time.time()

    # 1) Cargar KB (desde memoria / cache / GCS)
    t1 = time.time()
    chunks, vectors, metadata = get_kb()
    t2 = time.time()
    timings["load_kb"] = t2 - t1

    # 2) Embedding de la pregunta
    t3 = time.time()
    query_vec = embed_single(question)
    t4 = time.time()
    timings["embedding"] = t4 - t3

    # 3) Búsqueda vectorial (FAISS)
    t5 = time.time()
    results = search_top_k(query_vec, vectors, chunks, k=3)
    t6 = time.time()
    timings["vector_search"] = t6 - t5

    # Extraer chunks y scores
    top_chunks = [r["chunk"] for r in results]
    scores = [
        {"chunk_index": r["chunk_index"], "score": r["score"]}
        for r in results
    ]

    # Contexto para el prompt
    context = "\n\n".join(top_chunks)

    # 4) Construcción del prompt
    t7 = time.time()
    prompt = build_prompt(question, context)
    t8 = time.time()
    timings["prompt_build"] = t8 - t7

    # 5) Llamada al modelo generativo (usa Flash o Pro según la pregunta/contexto)
    t9 = time.time()
    answer = generate_answer(prompt, question, context)
    t10 = time.time()
    timings["generation"] = t10 - t9

    timings["total"] = t10 - t0

    return answer, top_chunks, scores, timings
