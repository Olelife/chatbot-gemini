import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.middleware import setup_middlewares
from core.genai_client import ensure_genai_client

from api.ask import router as ask_router
from api.debug import router as debug_router
from api.health import router as health_router
from api.reload import router as reload_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Startup")
    ensure_genai_client()
    yield
    logger.info("API Shutdown")


app = FastAPI(
    title="OleLife ChatBot Gemini API",
    description="API conversational bot",
    version="1.0.0",
    lifespan=lifespan
)
setup_middlewares(app)

app.include_router(ask_router)
app.include_router(debug_router)
app.include_router(reload_router)
app.include_router(health_router)
