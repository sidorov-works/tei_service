FROM --platform=linux/amd64 python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 libopenblas-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код и создаем папки
COPY . .
RUN mkdir -p logs models

# Скачиваем модель под root (имеем право писать в /app)
RUN python -m shared.utils.download_models

# Настройки окружения
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# Безопасный пользователь (после загрузки модели!)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HEALTHCHECK (проверяет localhost ВНУТРИ контейнера)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8260/health || exit 1