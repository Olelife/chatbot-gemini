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
        return data

    if not verify_slack_signature(request, body_str):
        return {"error": "invalid_signature"}

    event_id = data.get("event_id")
    event = data.get("event", {})

    # 🔥 evita duplicados
    if event_id in PROCESSED_EVENTS:
        logger.info(f"Ignored duplicate event: {event_id}")
        return {"ok": True}
    PROCESSED_EVENTS.add(event_id)

    event_type = event.get("type")
    user_id = event.get("user")
    channel = event.get("channel")
    text = event.get("text")
    thread_ts = event.get("thread_ts", event.get("ts"))  # HILO 👌

    if user_id == data["authorizations"][0]["user_id"]:
        return {"ok": True}  # evita responderte a ti mismo

    if event_type in ["app_mention", "message"]:
        slack_typing(channel, thread_ts)  # muestra escribiendo…

        background_tasks.add_task(
            handle_slack_question,
            text,
            user_id,
            channel,
            thread_ts,  # 👈 hilo para continuidad
            request
        )

    return {"ok": True}

def format_answer_to_blocks(answer: str, user: str):
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Hey <@{user}>! 👋\n\n{answer}"
            }
        },
        {"type": "divider"}
    ]

async def handle_slack_question(text, user_id, channel_id, thread_ts, request):
    result = process_question(
        question=text,
        session_id=f"slack-{user_id}-{thread_ts}",  # memoria por hilo
        country="br",
        username=user_id,
        request=request,
        background_tasks=None
    )

    blocks = format_answer_to_blocks(result["answer"], user_id)
    send_message_to_slack(channel_id, blocks, thread_ts)
