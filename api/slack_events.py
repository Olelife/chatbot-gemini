import logging
import hashlib
import hmac
import time
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from slack_sdk import WebClient
from core.config import settings
from api.ask import process_question
from services.slack_service import send_message_to_slack, slack_typing

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["Slack"])
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)

PROCESSED_EVENTS = set()
SLACK_SIGNING_SECRET = settings.SLACK_SIGNING_SECRET


def verify_slack_signature(request: Request, body: str):
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    slack_signature = request.headers.get("X-Slack-Signature")

    if abs(time.time() - int(timestamp)) > 300:
        return False

    base = f"v0:{timestamp}:{body}".encode()
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), base, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, slack_signature)


@router.post("/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    body_raw = await request.body()
    body_str = body_raw.decode("utf-8")
    data = await request.json()

    # Slack URL Verification
    if "challenge" in data:
        return {"challenge": data["challenge"]}

    if not verify_slack_signature(request, body_str):
        return {"error": "invalid_signature"}

    event_id = data.get("event_id")
    event = data.get("event", {})

    # 🔥 evita duplicados
    if event_id in PROCESSED_EVENTS:
        logger.info(f"Ignored duplicate event: {event_id}")
        return {"ok": True}
    PROCESSED_EVENTS.add(event_id)

    if event.get("type") == "app_mention":
        user_id = event.get("user")
        channel_id = event.get("channel")
        text = event.get("text")
        thread_ts = event.get("ts")  # 🧵 para responder en thread

        logger.info(f"@Mention by {user_id}: {text}")

        # ⏳ Inmediato → typing en hilo
        slack_typing(channel_id, thread_ts)

        # 👇 Lanzar procesamiento RAG/Gemini en background
        background_tasks.add_task(
            handle_slack_question,
            text,
            user_id,
            channel_id,
            thread_ts
        )

        return {"ok": True}

    return {"ok": True}


async def handle_slack_question(text, user_id, channel_id, thread_ts):
    result = process_question(
        question=text,
        session_id=f"slack-{user_id}",
        country="br",
        username=user_id,
        request=None,
        background_tasks=None,
    )

    send_message_to_slack(channel_id, result["answer"], thread_ts)
