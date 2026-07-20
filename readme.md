# TEI Service

TEI-совместимый сервис для векторизации текста и классификации.

Поддерживает два режима:
- **Encoder** — эмбеддинги текста через Sentence Transformers
- **Classifier** — классификация текста через sequence classification модели

## Быстрый старт

### Локальный запуск (Mac с Metal)

```bash
git clone <repository-url>
cd tei-service

python -m venv venv
source venv/bin/activate

pip install -r requirements.mps.txt

cp .env.frida.example .env.frida
# Отредактируйте .env.frida:
# SERVER_TYPE=encoder
# DEVICE=mps
# HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA

honcho start -f procfile -e .env.frida
```

### Локальный запуск (Linux)

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

python3 -m venv venv
source venv/bin/activate

# Для CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Для CUDA 12.1:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.cuda.txt

cp .env.example .env
# Укажите SERVER_TYPE=encoder или classifier
# Укажите DEVICE=cuda или cpu

uvicorn main:app --port 8262 --workers 1
```

## Запуск в Docker

### Требования

- Docker
- NVIDIA Container Toolkit (для GPU)

### Сборка и запуск

```bash
# Сборка базового образа (один раз, тяжело)
docker build -f dockerfile.base -t tei-base:latest .

# Запуск сервисов
docker compose up -d

# Проверка
curl http://localhost:8260/health
curl http://localhost:8265/health
```

### Остановка

```bash
docker compose down
```

### Полная очистка (удаление моделей и логов)

```bash
docker compose down -v
```

## Docker файлы

### dockerfile.base

Базовый образ с зависимостями. Собирается редко.

```dockerfile
# Базовый образ с Python 3.11
FROM python:3.11-slim

# ============================================
# 1. СИСТЕМНЫЕ ЗАВИСИМОСТИ
# ============================================

RUN apt-get update && apt-get install -y \
    curl \
    libopenblas-dev \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================
# 2. PYTHON ЗАВИСИМОСТИ (ВКЛЮЧАЯ TORCH)
# ============================================

COPY requirements.cuda.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.cuda.txt && \
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# ============================================
# 3. ПОЛЬЗОВАТЕЛЬ И ПРАВА
# ============================================

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app && \
    chmod 755 /app/models /app/logs

# ============================================
# 4. ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ============================================

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
```

### dockerfile.service

Сервисный образ с кодом. Собирается часто.

```dockerfile
# Наследуемся от базового образа, где уже установлены все зависимости
FROM tei-base:latest

# ============================================
# 1. КОПИРОВАНИЕ КОДА ПРИЛОЖЕНИЯ
# ============================================

COPY . .

# ============================================
# 2. ПРАВА НА ФАЙЛЫ
# ============================================

RUN chown -R appuser:appuser /app

# ============================================
# 3. ПЕРЕКЛЮЧЕНИЕ НА ПОЛЬЗОВАТЕЛЯ
# ============================================

USER appuser

# ============================================
# 4. HEALTHCHECK
# ============================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8260}/health || exit 1
```

### docker-compose.yml

```yaml
services:
  encoder_frida:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8260:8260"
    volumes:
      - model_cache:/app/models
      - logs_encoder:/app/logs
    env_file:
      - .env
      - .env.frida
    environment:
      - PORT=8260
      - DOCKER_ENV=true
    user: "1000:1000"
    container_name: encoder_frida
    restart: unless-stopped
    command: uvicorn main:app --port 8260 --host 0.0.0.0 --workers 1
    init: true
    shm_size: '2gb'
    mem_limit: 8g
    cpus: '4.0'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  classifier_pikabu:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8265:8265"
    volumes:
      - model_cache:/app/models
      - logs_classifier:/app/logs
    env_file:
      - .env
      - .env.pikabu
    environment:
      - PORT=8265
      - DOCKER_ENV=true
    user: "1000:1000"
    container_name: classifier_pikabu
    restart: unless-stopped
    command: uvicorn main:app --port 8265 --host 0.0.0.0 --workers 1
    init: true
    shm_size: '2gb'
    mem_limit: 8g
    cpus: '4.0'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  model_cache:
    name: tei_model_cache
  logs_encoder:
    name: tei_logs_encoder
  logs_classifier:
    name: tei_logs_classifier
```

## Эндпоинты

### GET /health

Проверка здоровья сервиса.

- `200 OK` — сервис готов
- `503 Service Unavailable` — сервис не готов

### GET /info

Информация о загруженной модели.

### POST /embed (только в режиме энкодера)

Получение эмбеддингов текста.

**Request**
```json
{
  "inputs": ["текст1", "текст2"]
}
```

**Response**
```json
[
  [0.123, -0.456, 0.789, ...],
  [0.321, -0.654, 0.987, ...]
]
```

### POST /predict (только в режиме классификатора)

Классификация текста.

**Request**
```json
{
  "inputs": ["текст1", "текст2"]
}
```

**Response**
```json
[
  [{"label": "toxic", "score": 0.98}],
  [{"label": "toxic", "score": 0.01}]
]
```

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| SERVER_TYPE | encoder или classifier | encoder |
| HUGGING_FACE_MODEL_NAME | Имя модели | ai-forever/FRIDA |
| DEVICE | cpu, cuda, mps | cpu |
| MAX_MODEL_BATCH_SIZE | Размер батча | 32 |
| MAX_SERVICE_BATCH_SIZE | Макс. размер батча для эндпоинта | 128 |
| MAX_TEXT_LENGTH | Макс. длина одного текста | 10000 |
| MAX_TOTAL_BATCH_LENGTH | Макс. суммарная длина текстов в батче | 500000 |
| INTERNAL_API_SECRET | Ключ аутентификации | не задан |
| REQUIRE_AUTH | Требовать аутентификацию | false |

## Устранение неполадок

### Просмотр логов

```bash
docker compose logs -f
docker compose logs -f encoder_frida
docker compose logs -f classifier_pikabu
```

### Проверка GPU

```bash
docker exec encoder_frida nvidia-smi
```

### Очистка

```bash
# Остановка
docker compose down

# Полная очистка (удаление томов)
docker compose down -v

# Очистка кэша Docker
docker system prune -a
```

## Лицензия

MIT