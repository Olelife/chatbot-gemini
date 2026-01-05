import logging
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.middleware import setup_middlewares
from api.ask import router as ask_router
from api.debug import router as debug_router
from api.health import router as health_router
from api.admin import router as admin_router
from core.genai_client import init_genai_client
from api.slack_events import router as slack_events_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variable global para trackear estado
genai_ready = False

async def init_genai_async():
    """Inicializa GenAI en background sin bloquear el startup"""
    global genai_ready
    try:
        logger.info("🔧 Initializing GenAI client...")
        # Ejecutar en thread separado para no bloquear
        await asyncio.to_thread(init_genai_client)
        genai_ready = True
        logger.info("✅ GenAI client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize GenAI client: {e}", exc_info=True)
        genai_ready = False
        # NO re-raise - permitir que el servidor siga corriendo

@asynccontextmanager
async def lifespan(app: FastAPI):
    port = os.environ.get("PORT", "8080")
    logger.info("=" * 60)
    logger.info(f"🚀 API Startup - Port: {port}")
    logger.info("=" * 60)
    
    # Inicializar en background - NO esperar ni bloquear
    asyncio.create_task(init_genai_async())
    
    # El servidor está listo para aceptar requests inmediatamente
    logger.info("✅ Server ready - Background initialization in progress...")
    
    yield
    
    logger.info("🛑 API Shutdown")

app = FastAPI(
    title="OleLife ChatBot Gemini API",
    description="API conversational bot",
    version="1.0.0",
    lifespan=lifespan
)

setup_middlewares(app)
app.include_router(health_router)  # Health primero para que responda rápido
app.include_router(ask_router)
app.include_router(debug_router)
app.include_router(admin_router)
app.include_router(slack_events_router)

# Root endpoint para verificar estado
@app.get("/")
async def root():
    return {
        "service": "OleLife ChatBot Gemini API",
        "version": "1.0.0",
        "status": "running",
        "genai_ready": genai_ready
    }
