import logging
from slack_sdk import WebClient
from core.config import settings

logger = logging.getLogger(__name__)
client = WebClient(token=settings.SLACK_BOT_TOKEN)

def publish_home_tab_br(user_id: str):
    try:
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🤖 Bem-vindo ao Olé Assistant!"}
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Sou seu assistente de seguros de vida. Me mencione aqui ou em qualquer canal!"
                        }
                    },
                    {"type": "divider"},
                    {
                        "type": "actions",
                        "elements": [{
                            "type": "button",
                            "text": {"type": "plain_text", "text": "❓ Fazer uma pergunta"},
                            "action_id": "ask_button"
                        }]
                    }
                ]
            }
        )

        logger.info(f"[SLACK] Home tab published for {user_id}")

    except Exception as e:
        logger.error(f"Home tab error: {e}")
