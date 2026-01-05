import logging
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API Startup iniciando...")
    try:
        logger.info("Inicializando GenAI client...")
        init_genai_client()
        logger.info("✅ GenAI client inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando GenAI client: {str(e)}", exc_info=True)
        # NO hagas raise aquí, permite que la app inicie
        logger.warning("⚠️ App iniciará sin GenAI client")
    
    logger.info("✅ API Startup completado")
    yield
    logger.info("🛑 API Shutdown")

app = FastAPI(
    title="OleLife ChatBot Gemini API",
    description="API conversational bot",
    version="1.0.0",
    lifespan=lifespan
)

setup_middlewares(app)

# Routers
app.include_router(health_router)  # Health PRIMERO
app.include_router(ask_router)
app.include_router(debug_router)
app.include_router(admin_router)
app.include_router(slack_events_router)

logger.info(f"🌐 Configurado para escuchar en puerto: {os.getenv('PORT', '8080')}")
