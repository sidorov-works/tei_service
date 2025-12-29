# FROM --platform=linux/amd64 python:3.11-slim
FROM --platform=linux/arm64 python:3.11-slim

# Минимум: только curl для healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код и создаем папки
COPY . .
RUN mkdir -p logs models

# Настройки окружения:
    # выводит логи сразу в консоль Docker, без задержек
ENV PYTHONUNBUFFERED=1 \  
    # останавливает внутреннюю многопоточность PyTorch, которая ломает работу в контейнере 
    OMP_NUM_THREADS=1 \     
    # отключает параллельные токенизаторы, иначе будут deadlock при вызовах /count_tokens        
    TOKENIZERS_PARALLELISM=false

# Безопасный пользователь
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8260/health || exit 1