import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from core.config import settings
from utils.logger import generate_trace_id, log_error, log_info

#logger = logging.getLogger(__name__)
trace_id = generate_trace_id()

# Slack Bot Client
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)

#def send_message_to_slack(channel: str, text: str):
#    """
#    Envía un mensaje al canal en Slack usando chat.postMessage
#    """
#    try:
#        response = slack_client.chat_postMessage(
#            channel=channel,
#            text=text
#        )
#        logger.info(f"Message sent to Slack channel {channel}: {response.data}")
#    except SlackApiError as e:
#        logger.error(f"Slack API Error: {e.response['error']}")

def slack_typing(channel: str, thread_ts: str | None = None):
    slack_client.chat_postMessage(
        channel=channel,
        text="⏳ Estou analisando sua pergunta...",
        thread_ts=thread_ts
    )


def send_message_to_slack(channel, blocks, thread_ts=None, trace_id=None):
    """
    #    Envía un mensaje al canal en Slack usando chat.postMessage
    #    """
    try:
        response = slack_client.chat_postMessage(
            channel=channel,
            blocks=blocks,
            text="Nueva respuesta de Olé Assistant",
            thread_ts=thread_ts
        )
        log_info(
            f"Message sent to Slack channel {channel}: {response.data}",
            trace_id=trace_id,
            channel=channel,
            thread_ts=thread_ts
        )
    except SlackApiError as e:
#       #logger.error(f"Slack API Error: {e.response['error']}")
        log_error(
            "Slack API published failed",
            trace_id=trace_id,
            error=f"Slack API Error: {e.response['error']}"
        )
