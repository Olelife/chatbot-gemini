import re

def markdown_to_slack(text: str) -> str:
    # Negritas: **texto** → *texto*
    text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)

    # También limpiamos triples asteriscos por si aparecen: ***texto***
    text = re.sub(r"\*{3}(.*?)\*{3}", r"*\1*", text)

    # Heading Markdown tipo: ## Título → *Título:* (Slack-friendly)
    text = re.sub(r"^##+\s*(.*)", r"*\1:*", text, flags=re.MULTILINE)

    return text

def sanitize_answer(raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    text = "\n".join(lines)

    # Normalizar viñetas Slack
    text = re.sub(r'\n\s*[-*]\s+', "\n• ", text)

    # 🔥 Convertir Markdown a mrkdwn de Slack
    text = markdown_to_slack(text)

    return text