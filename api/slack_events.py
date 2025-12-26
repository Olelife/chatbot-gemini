import logging
import hashlib
import hmac
import time
from fastapi import APIRouter, Request, HTTPException
from slack_sdk import WebClient
from core.config import settings
from api.ask import process_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["Slack"])
client = WebClient(token=settings.SLACK_BOT_TOKEN)

@router.post("/events")
async def slack_events(request: Request):
    # ---- Verificar firma Slack ----
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    slack_signature = request.headers.get("X-Slack-Signature")

    if abs(time.time() - int(timestamp)) > 60 * 5:
        raise HTTPException(status_code=400, detail="Timestamp expired")

    body = await request.body()
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    signature = 'v0=' + hmac.new(
        settings.SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, slack_signature):
        raise HTTPException(status_code=403, detail="Signature mismatch")

    data = await request.json()

    # ---- Challenge Request (evento inicial) ----
    if "challenge" in data:
        return {"challenge": data["challenge"]}

    event = data.get("event", {})
    text = event.get("text", "")
    user_id = event.get("user")
    channel = event.get("channel")

    # Evitar responderse a sí mismo
    if event.get("bot_id"):
        return {"status": "ignored"}

    # Detectar mención
    logger.info(f"Received mention: {text}")

    result = process_question(
        question=text.replace(f"<@{user_id}>", "").strip(),
        session_id=f"slack-{user_id}",
        country="br",
        username=user_id,
        request=request,
        background_tasks=None,
    )

    client.chat_postMessage(
        channel=channel,
        text=result["answer"]
    )

    return {"status": "ok"}
