import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from core.config import settings

logger = logging.getLogger(__name__)

# Slack Bot Client
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)

def send_message_to_slack(channel: str, text: str):
    """
    Envía un mensaje al canal en Slack usando chat.postMessage
    """
    try:
        response = slack_client.chat_postMessage(
            channel=channel,
            text=text
        )
        logger.info(f"Message sent to Slack channel {channel}: {response.data}")
    except SlackApiError as e:
        logger.error(f"Slack API Error: {e.response['error']}")

def slack_typing(channel: str, thread_ts: str | None = None):
    slack_client.chat_postMessage(
        channel=channel,
        text="typing...",
        thread_ts=thread_ts
    )


def send_message_to_slack(channel, text, thread_ts=None):
    slack_client.chat_postMessage(
        channel=channel,
        text=text,
        thread_ts=thread_ts
    )
