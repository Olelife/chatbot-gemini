import pytest
import httpx
import time

API_URL = "http://localhost:8080/ask"

# Valores máximos permitidos por etapa
LIMITS = {
    "embedding": 2.0,
    "vector_search": 0.05,
    "prompt_build": 0.05,
    "generation": 6.0,
    "db_log": 0.5,
    "total": 8.0
}

@pytest.mark.parametrize("question", [
    "¿Cuál es tu nombre?",
    "¿Qué coberturas tiene el producto?",
    "¿Cuál es la edad máxima?",
])
def test_rag_timings(question):
    headers = {
        "x-username": "test-user",
        "x-session-id": "test-session",
        "x-country": "mx",
    }

    response = httpx.post(API_URL, json={"question": question}, headers=headers)
    assert response.status_code == 200

    data = response.json()
    timings = data["timings"]

    # Validar presencia de métricas
    for key in LIMITS.keys():
        assert key in timings, f"Missing timing metric: {key}"

    # Validar que cada etapa esté dentro de límites razonables
    for key, limit in LIMITS.items():
        assert timings[key] < limit, f"{key} too slow: {timings[key]}s (limit {limit}s)"


def test_performance_under_load():
    """Envía 10 requests secuenciales y mide promedio."""
    headers = {
        "x-username": "test-user",
        "x-session-id": "load-test",
        "x-country": "mx",
    }

    durations = []

    for i in range(10):
        start = time.time()
        resp = httpx.post(API_URL, json={"question": "¿Qué coberturas hay?"}, headers=headers)
        assert resp.status_code == 200
        durations.append(time.time() - start)

    avg_duration = sum(durations) / len(durations)

    assert avg_duration < 5.5, f"Average response time too slow: {avg_duration}s"
