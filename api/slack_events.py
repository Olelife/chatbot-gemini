import logging
import hashlib
import hmac
import time
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from slack_sdk import WebClient

from api.slack_home import publish_home_tab_br
from core.config import settings
from api.ask import process_question
from services.slack_service import send_message_to_slack, slack_typing
from utils.markdown import sanitize_answer
from utils.logger import log_info, log_error, generate_trace_id

#logger = logging.getLogger(__name__)
trace_id = generate_trace_id()

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
    event_type = event.get("type")
    user_id = event.get("user")
    channel = event.get("channel")
    text = event.get("text")
    thread_ts = event.get("thread_ts", event.get("ts"))  # HILO 👌

    # 🔥 evita duplicados
    if event_id in PROCESSED_EVENTS:
        log_info(
            f"SLACK PROCESSED EVENT",
            trace_id=trace_id,
            event_type=event_type,
            user_id=user_id,
            channel=channel,
            thread_ts=thread_ts,
            text=f"SLACK Ignored duplicate event: {event_id}",
        )
        return {"ok": True}
    PROCESSED_EVENTS.add(event_id)

    if user_id == data["authorizations"][0]["user_id"]:
        return {"ok": True}  # evita responderte a ti mismo

    if event_type in ["app_mention", "message"]:
        log_info(
            f"SLACK MENTION EVENT",
            trace_id=trace_id,
            event_type=event_type,
            user_id=user_id,
            channel=channel,
            thread_ts=thread_ts,
            text=text,
        )

        question_intent = is_real_question(text)

        if question_intent:
            slack_typing(channel, thread_ts)
            background_tasks.add_task(
                handle_slack_question,
                text,
                user_id,
                channel,
                thread_ts,
                request
            )
        else:
            background_tasks.add_task(
                handle_slack_question,
                text,
                user_id,
                channel,
                None,
                request
            )

    if event_type == "app_home_opened":
        user_id = event.get("user")
        log_info(
            f"[SLACK] Home opened by {user_id}",
            trace_id=trace_id,
            channel=channel,
            thread_ts=thread_ts
        )
        publish_home_tab_br(user_id)

    return {"ok": True}

def format_answer_to_blocks(answer: str, user: str):
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Hey <@{user}>! 👋\n\n{sanitize_answer(answer)}"
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
    log_info(
        "Message sent to Slack",
        trace_id=trace_id,
        channel=channel_id,
        thread_ts=thread_ts
    )
    try:
        blocks = format_answer_to_blocks(result["answer"], user_id)
        log_info(
            "Formatted message sent to Slack",
            trace_id=trace_id,
            channel=channel_id,
            thread_ts=thread_ts
        )
        send_message_to_slack(channel_id, blocks, thread_ts)
    except Exception as e:
        log_error(
            "Slack processing failed",
            trace_id=trace_id,
            error=str(e)
        )

def is_real_question(text: str) -> bool:
    """
    Determina si el mensaje requiere procesamiento profundo.
    """
    text = text.lower().strip()
    is_question = "?" in text

    greetings = ["hola", "oi", "olá", "ola", "hey", "buenas", "bom dia", "boa tarde"]

    only_greeting = any(text == g for g in greetings)

    # es pregunta real
    if is_question:
        return True

    # si solo es greeting: no es pregunta
    if only_greeting:
        return False

    # frases de una sola intención sin pregunta
    if len(text.split()) < 3:
        return False

    return True