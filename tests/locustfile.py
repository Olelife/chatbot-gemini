from locust import HttpUser, task, between

class BasicChatbotUser(HttpUser):
    wait_time = between(1, 30)

    @task
    def ask_simple_question(self):
        payload = {
            "question": "¿Cuál es la edad mínima para contratar?"
        }

        headers = {
            "x-username": "test-user",
            "x-country": "mx",
            "x-session-id": "session123"
        }

        # Usa catch_response para poder marcar success/failure sin error
        with self.client.post(
            "/ask",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:

            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text}")
            else:
                # Puedes hacer validaciones de contenido
                try:
                    data = response.json()
                    if "answer" not in data:
                        response.failure("No 'answer' field in response")
                    else:
                        response.success()
                except Exception as e:
                    response.failure(f"JSON decode error: {e}")
