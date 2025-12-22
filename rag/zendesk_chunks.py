def convert_zendesk_articles_to_chunks(articles: list[dict]) -> list[str]:
    """
    Convierte artículos Zendesk en chunks semánticos completos para RAG.
    No hacemos splitting por párrafo; cada artículo = chunk.
    """

    chunks = []

    for art in articles:
        block = f"""
## ZENDESK_BR_ARTICLE
TITLE: {art['title']}
NAME: {art['name']}
URL: {art['url']}

CONTENT:
{art['body']}
        """
        chunks.append(block.strip())

    return chunks
