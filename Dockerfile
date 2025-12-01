FROM python:3.10-slim

WORKDIR /app

# Copiar archivos primero
COPY requirements.txt .

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer puerto
EXPOSE 8080

# CORRECCIÓN: Sin '=' en los argumentos
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
