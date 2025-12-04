# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from core.genai_client import init_genai_client
from rag.knowledge import get_kb
from api.ask import router as ask_router
from api.debug import router as debug_router
from api.health import router as health_router
from api.reload import router as reload_router
from api.middleware import setup_middlewares

from rag.embeddings import embed_single

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "initialized", False) is False:
        app.state.initialized = True
        logger.info("🚀 API starting up...")

        # 1. Inicializar GenAI client
        client = init_genai_client()

        # 2. Precargar KB y embeddings
        chunks, vectors, metadata = get_kb()
        # Warmup: embedding + búsqueda FAISS
        vec = embed_single("warmup")
        from rag.search import search_top_k
        search_top_k(vec, vectors, chunks)
        logger.info(f"✓ KB loaded: {len(chunks)} chunks, emb_dim={vectors.shape[1]}")

        # 3. Warmup del embedding model
        try:
            _ = embed_single("warmup")
            logger.info("✓ Embedding model warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    yield

    logger.info("🛑 API shutting down...")

app = FastAPI(
    title="OleLife RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Middlewares
setup_middlewares(app)

# Routers
app.include_router(ask_router)
app.include_router(debug_router)
app.include_router(health_router)
app.include_router(reload_router)
