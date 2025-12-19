import time
import statistics
import asyncio
import httpx
import pytest

API_URL = "http://localhost:8080/ask"

HEADERS = {
    "x-username": "perf-test",
    "x-country": "mx",
    "x-session-id": "perf-session"
}

QUESTION = {"question": "¿Qué edades están permitidas?"}


# ================
# 1) Tiempo máximo por request
# ================
@pytest.mark.asyncio
async def test_response_time_under_30_seconds():
    async with httpx.AsyncClient(timeout=30) as client:
        start = time.time()
        r = await client.post(API_URL, json=QUESTION, headers=HEADERS)
        elapsed = time.time() - start

        assert r.status_code == 200
        print(r.json())
        assert elapsed < 30, f"Respuesta demasiado lenta: {elapsed:.2f}s"


# ================
# 2) Throughput — 20 requests secuenciales
# ================
@pytest.mark.asyncio
async def test_throughput_sequential_20_requests():
    async with httpx.AsyncClient(timeout=150) as client:
        times = []

        for _ in range(20):
            start = time.time()
            r = await client.post(API_URL, json=QUESTION, headers=HEADERS)
            elapsed = time.time() - start
            times.append(elapsed)
            assert r.status_code == 200

        avg = statistics.mean(times)

        print("\n--- PERFORMANCE (20 sequential requests) ---")
        print(f"Promedio: {avg:.3f}s")
        print(f"p95: {statistics.quantiles(times, n=20)[-1]:.3f}s")

        assert avg < 12.5, "Promedio muy alto"


# ================
# 3) Concurrencia — 20 usuarios simultáneos
# ================
async def make_request(client):
    start = time.time()
    r = await client.post(API_URL, json=QUESTION, headers=HEADERS)
    elapsed = time.time() - start
    return r.status_code, elapsed


@pytest.mark.asyncio
async def test_concurrency_20_users():
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [make_request(client) for _ in range(20)]
        results = await asyncio.gather(*tasks)

        times = []
        for status, elapsed in results:
            assert status == 200
            times.append(elapsed)

        avg = statistics.mean(times)
        p95 = statistics.quantiles(times, n=20)[-1]
        p99 = max(times)

        print("\n--- CONCURRENCY 20 USERS ---")
        print(f"Avg:  {avg:.3f}s")
        print(f"P95:  {p95:.3f}s")
        print(f"P99:  {p99:.3f}s")

        assert p95 < 14.0, "p95 muy alto con 20 usuarios"
