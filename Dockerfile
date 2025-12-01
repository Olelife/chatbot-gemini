FROM python:3.10-slim

WORKDIR /app

# Copiar requirements primero para aprovechar caché
COPY requirements.txt .

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Puerto
EXPOSE 8080

# Comando para iniciar
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info
