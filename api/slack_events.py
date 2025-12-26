import logging
import hashlib
import hmac
import time
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from slack_sdk import WebClient
from core.config import settings
from api.ask import process_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["Slack"])
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)

def verify_slack_signature(request: Request, body: str):
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")

    if abs(time.time() - int(timestamp)) > 300:
        raise HTTPException(status_code=403, detail="Timestamp expired")

    base = f"v0:{timestamp}:{body}".encode()
    secret = settings.SLACK_SIGNING_SECRET.encode()

    computed = "v0=" + hmac.new(secret, base, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

@router.post("/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()
    body = raw_body.decode()

    verify_slack_signature(request, body)
    payload = await request.json()

    # Slack URL Verification challenge
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})
    event_type = event.get("type")

    # Ignorar eventos no relevantes (typing, bot messages, etc.)
    if event.get("bot_id"):
        return {"ok": True}

    user_id = event.get("user")
    channel_id = event.get("channel")

    # Detectar fuente del mensaje
    if event_type == "app_mention":
        text = event.get("text", "").replace(f"<@{settings.SLACK_BOT_USER_ID}>", "").strip()
        logger.info(f"App Mention from {user_id} -> {text}")
    elif event_type == "message" and event.get("channel_type") == "im":
        text = event.get("text")
        logger.info(f"DM from {user_id} -> {text}")
    else:
        return {"ok": True}

    country = "br"  # Por defecto para Slack
    session_id = f"slack-{user_id}"

    logger.info(
        msg=f"Question {text} with event type {event_type} amd channel {channel_id}"
    )

    result = process_question(
        text,
        session_id,
        country,
        user_id,
        request,
        background_tasks,
    )

    # Enviar respuesta a Slack
    slack_client.chat_postMessage(
        channel=channel_id,
        text=result["answer"]
    )

    return {"ok": True}
