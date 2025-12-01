
##RUN pip install --no-cache-dir -r requirements.txt
##EXPOSE 8080
##CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]

FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
