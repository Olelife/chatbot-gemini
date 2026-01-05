from fastapi import APIRouter
import logging

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)

@router.get("")
async def health_check():
    """Health check que SIEMPRE responde, incluso si otros servicios fallan"""
    logger.info("Health check llamado")
    return {
        "status": "healthy",
        "service": "olelife-chatbot"
    }

@router.get("/ready")
async def readiness_check():
    """Readiness check - verifica que los servicios estén listos"""
    from core.genai_client import genai_client  # o como lo tengas
    
    checks = {
        "genai": False
    }
    
    try:
        # Verifica si el cliente está inicializado
        if genai_client is not None:
            checks["genai"] = True
    except:
        pass
    
    status = "ready" if all(checks.values()) else "not_ready"
    
    return {
        "status": status,
        "checks": checks
    }
