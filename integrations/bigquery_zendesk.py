from google.cloud import bigquery

def fetch_zendesk_articles_br() -> list[dict]:
    """
    Consulta artículos de Zendesk (BR) almacenados en BigQuery.
    """

    client = bigquery.Client()

    QUERY = """
        SELECT name, title, body, url
        FROM `olelifetech.raw_zone.zendesk_br_hc_articles`
        WHERE locale = 'pt-br'
    """

    rows = client.query(QUERY).result()

    articles = []
    for row in rows:
        articles.append({
            "name": row.name,
            "title": row.title,
            "body": row.body,
            "url": row.url,
        })

    return articles
