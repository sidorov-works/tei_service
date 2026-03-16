# Encoder Service

Сервис векторизации текста, предоставляющий TEI-совместимый HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Содержание
- [Архитектура](#архитектура)
- [API Эндпоинты](#api-эндпоинты)
- [Модели данных](#модели-данных)
- [Управление таймаутами](#управление-таймаутами)
- [Лимиты и валидация](#лимиты-и-валидация)
- [Очередь и многопоточность](#очередь-и-многопоточность)
- [Аутентификация](#аутентификация)
- [Установка и запуск](#установка-и-запуск)
  - [Локальный запуск](#локальный-запуск)
  - [Запуск с honcho (Procfile)](#запуск-с-honcho-procfile)
  - [Запуск в Docker](#запуск-в-docker)
  - [Запуск нескольких экземпляров](#запуск-нескольких-экземпляров)
- [Конфигурация](#конфигурация)
  - [Переменные окружения](#переменные-окружения)
  - [Примеры .env файлов](#примеры-env-файлов)
- [Офлайн-режим](#офлайн-режим)
- [Мониторинг](#мониторинг)
- [Безопасность](#безопасность)
- [Клиентская библиотека (Encoder Client)](#клиентская-библиотека-encoder-client)

### Архитектура

Сервис построен с учетом следующих ключевых требований:

- **Thread-safe работа с моделями** — SentenceTransformer модели не являются потокобезопасными. В сервисе реализована архитектура "очередь + воркер":
  - FastAPI эндпоинты принимают запросы и ставят их в очередь `input_queue`
  - Один выделенный воркер (`ModelWorker`) последовательно обрабатывает задачи, имея эксклюзивный доступ к модели
  - Диспетчер результатов (`ResultDispatcher`) направляет ответы ожидающим запросам через `asyncio.Future`
  - Все операции с Future, которые могут вызвать колбэки, выносятся в отдельные потоки через `ThreadPoolExecutor`

- **Батчевая обработка** — поддержка пакетного кодирования нескольких текстов за один запрос для оптимальной производительности.

- **TEI-совместимый API** — эндпоинты соответствуют спецификации Text Embeddings Inference, что позволяет использовать сервис как замену TEI.

- **Аутентификация по секретному заголовку** — все рабочие эндпоинты защищены статическим API-ключом, передаваемым в заголовке `Authorization: Bearer <token>`.

- **Локальное хранение моделей** — модели скачиваются один раз и сохраняются локально в `models/sentence-transformers/`. При наличии локальной копии сервис работает полностью офлайн.

- **Защита от перегрузок** — очередь запросов с максимальным размером (1000 задач) и валидация размера входящих данных. При заполнении очереди > 900 новые запросы получают 503 Service Unavailable.

### API Эндпоинты

**Важно:** Все POST эндпоинты требуют аутентификации через заголовок `Authorization: Bearer <api-key>`.

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/info` | GET | Информация о модели (не требует аутентификации) |
| `/embed` | POST | Получение эмбеддингов (одиночный или пакетный запрос) |
| `/tokenize` | POST | Подсчет токенов (одиночный или пакетный запрос) |
| `/health` | GET | Проверка здоровья сервиса (не требует аутентификации) |

#### Информация о модели (`/info`)

```bash
curl http://localhost:8260/info
```

```json
{
    "model_id": "deepvk/USER2-base",
    "max_input_length": 8192,
    "dimension": 768,
    "status": "operational"
}
```

#### Получение эмбеддингов (`/embed`)

**Одиночный текст:**
```bash
curl -X POST http://localhost:8260/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "inputs": "Проблема с доступом к почте",
    "prompt_name": "query",
    "normalize": true
  }'
```

```json
{
    "embedding": [0.123, -0.456, ...]
}
```

**Пакетный запрос:**
```bash
curl -X POST http://localhost:8260/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "inputs": ["текст1", "текст2"],
    "prompt_name": "document",
    "normalize": true
  }'
```

```json
{
    "embeddings": [[0.123, -0.456, ...], [0.789, -0.321, ...]]
}
```

#### Подсчет токенов (`/tokenize`)

**Одиночный текст:**
```bash
curl -X POST http://localhost:8260/tokenize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "inputs": "Пример текста для подсчета токенов",
    "add_special_tokens": true
  }'
```

```json
{
    "tokens_count": 8
}
```

**Пакетный запрос:**
```bash
curl -X POST http://localhost:8260/tokenize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "inputs": ["текст1", "текст2"],
    "add_special_tokens": true
  }'
```

```json
{
    "tokens_counts": [5, 8]
}
```

#### Проверка здоровья (`/health`)

```bash
curl http://localhost:8260/health
```

```json
{
    "status": "healthy",
    "encoder_loaded": true,
    "queue_size": 0,
    "active_requests": 0,
    "service_available": true,
    "max_timeout": 120
}
```

### Модели данных

#### EmbedRequest

```python
{
    "inputs": "string" | ["string1", "string2"],  # текст или список текстов
    "prompt_name": "query",                        # опционально, "query" или "document"
    "normalize": true,                             # опционально, нормализация вектора
    "truncate": true                               # опционально, обрезать до max_length
}
```

#### TokenizeRequest

```python
{
    "inputs": "string" | ["string1", "string2"],  # текст или список текстов
    "add_special_tokens": true,                    # опционально, добавлять спец. токены
    "truncate": true                                # опционально, обрезать до max_length
}
```

Валидация:
- Количество текстов в батче ≤ `MAX_BATCH_SIZE` (256)
- Каждый текст не пустой и ≤ `MAX_TEXT_LENGTH` (10000 символов)
- Суммарная длина всех текстов в батче ≤ `MAX_TOTAL_BATCH_LENGTH` (500000 символов)

### Управление таймаутами

Сервис не поддерживает передачу таймаутов в теле запроса (в соответствии со спецификацией TEI). Таймауты настраиваются только на стороне сервера:

- `ENCODER_BASE_TIMEOUT` (15с) — для одиночных запросов (`/embed` с одним текстом, `/tokenize` с одним текстом)
- `ENCODER_BATCH_TIMEOUT` (60с) — для пакетных запросов (`/embed` с несколькими текстами, `/tokenize` с несколькими текстами)
- `MAX_SERVICE_TIMEOUT` (120с) — жесткий потолок, сервис никогда не ждет дольше

### Очередь и многопоточность

Сервис использует асинхронную очередь для обработки запросов:

- **input_queue**: максимальный размер 1000 задач
- При заполнении очереди > 900 новые запросы получают 503 Service Unavailable
- **Один воркер** последовательно обрабатывает задачи, имея эксклюзивный доступ к модели
- **Диспетчер результатов** направляет ответы от воркера к ожидающим Future
- Для избежания блокировки event loop, операции с Future (`set_result`, `set_exception`) выполняются в отдельных потоках через `ThreadPoolExecutor`

### Аутентификация

Сервис использует аутентификацию по секретному заголовку (API-ключ). Клиент передает ключ в заголовке `Authorization: Bearer <api-key>`.

```python
# В main.py используется require_header_secret
from shared.auth_service import require_header_secret as require_auth
```

Ключ задается через переменную окружения `INTERNAL_API_SECRET`:

```bash
INTERNAL_API_SECRET=your-32-char-secret-key-minimum
```

Пример запроса с аутентификацией:
```bash
curl -X POST http://localhost:8260/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-32-char-secret-key-minimum" \
  -d '{"inputs": "пример текста"}'
```

### Установка и запуск

#### Локальный запуск

```bash
# Клонирование репозитория
git clone <repository-url>
cd encoder-service

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Создание .env файла с конфигурацией
cp .env.example .env
# Отредактируйте .env под вашу модель

# Запуск сервиса (ВАЖНО: всегда с 1 воркером!)
uvicorn encoder_service.main:app \
  --host 0.0.0.0 \
  --port 8260 \
  --workers 1 \
  --lifespan on \
  --timeout-graceful-shutdown 10
```

#### Запуск с honcho (Procfile)

Для разработки удобно использовать honcho:

```
# Procfile
encoder_service: uvicorn encoder_service.main:app \
  --port ${ENCODER_SERVICE_PORT} \
  --workers 1 \
  --lifespan on \
  --timeout-graceful-shutdown 10
```

Запуск:
```bash
# Для модели FRIDA
honcho start -f Procfile -e .env.frida

# Для модели deepvk
honcho start -f Procfile -e .env.deepvk
```

#### Запуск в Docker

##### Базовый образ (dockerfile.base)

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    libopenblas-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.windows.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.windows.txt

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
```

##### Сервисный образ (dockerfile.service)

```dockerfile
FROM encoder-base:latest

COPY . .

RUN mkdir -p logs models && \
    useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8260}/health || exit 1
```

Сборка:
```bash
docker build -f dockerfile.base -t encoder-base:latest .
docker build -f dockerfile.service -t encoder-service:latest .
```

#### Запуск нескольких экземпляров

```yaml
# docker-compose.yml
services:
  encoder-frida:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8260:8260"
    volumes:
      - ./models:/app/models
      - ./logs/encoder-frida:/app/logs
    env_file:
      - .env
      - .env.frida
    environment:
      - PORT=8260
    container_name: encoder_frida
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8260 --host 0.0.0.0 --workers 1
      --lifespan on --timeout-graceful-shutdown 15
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  encoder-deepvk:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8261:8261"
    volumes:
      - ./models:/app/models
      - ./logs/encoder-deepvk:/app/logs
    env_file:
      - .env
      - .env.deepvk
    environment:
      - PORT=8261
    container_name: encoder_deepvk
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8261 --host 0.0.0.0 --workers 1
      --lifespan on --timeout-graceful-shutdown 15
```

Запуск:
```bash
docker-compose up -d
```

### Конфигурация

#### Переменные окружения

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| **Обязательные** |
| `INTERNAL_API_SECRET` | API-ключ для аутентификации | **Нет значения** |
| **Идентификация** |
| `ENCODER_NAME` | Уникальное имя экземпляра | "frida" |
| **Модель** |
| `HUGGING_FACE_MODEL_NAME` | Название модели | "ai-forever/FRIDA" |
| `DEVICE` | Устройство ("cpu", "cuda", "mps") | "cpu" |
| `QUERY_PREFIX` | Префикс для запросов | "search_query: " |
| `DOCUMENT_PREFIX` | Префикс для документов | "search_document: " |
| **Таймауты** |
| `ENCODER_BASE_TIMEOUT` | Для одиночных запросов (сек) | 15 |
| `ENCODER_BATCH_TIMEOUT` | Для batch запросов (сек) | 60 |
| `MAX_SERVICE_TIMEOUT` | Жесткий потолок (сек) | 120 |
| **Лимиты** |
| `MAX_BATCH_SIZE` | Макс. текстов в батче | 256 |
| `MAX_TEXT_LENGTH` | Макс. длина текста (символов) | 10000 |
| `MAX_TOTAL_BATCH_LENGTH` | Макс. суммарная длина | 500000 |
| **Rate limiting** |
| `RATE_LIMIT_INFO` | Лимит для /info, /health | "500/minute" |
| `RATE_LIMIT_ENCODE` | Лимит для /embed (одиночные) | "500/minute" |
| `RATE_LIMIT_ENCODE_BATCH` | Лимит для /embed (batch) | "150/minute" |
| `RATE_LIMIT_COUNT_TOKENS` | Лимит для /tokenize (одиночные) | "600/minute" |
| `RATE_LIMIT_COUNT_TOKENS_BATCH` | Лимит для /tokenize (batch) | "200/minute" |
| **Пути** |
| `LOG_PATH` | Директория для логов | "logs" |
| `MODEL_PATH` | Директория для моделей | "models/sentence-transformers" |

#### Примеры .env файлов

**Общий .env:**
```bash
INTERNAL_API_SECRET=your-32-char-secret-key-minimum

ENCODER_BASE_TIMEOUT=15
ENCODER_BATCH_TIMEOUT=60
MAX_SERVICE_TIMEOUT=120

MAX_BATCH_SIZE=256
MAX_TEXT_LENGTH=10000
MAX_TOTAL_BATCH_LENGTH=500000

RATE_LIMIT_INFO=500/minute
RATE_LIMIT_ENCODE=500/minute
RATE_LIMIT_ENCODE_BATCH=150/minute
RATE_LIMIT_COUNT_TOKENS=600/minute
RATE_LIMIT_COUNT_TOKENS_BATCH=200/minute
```

**.env.frida:**
```bash
ENCODER_NAME=frida
HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
DEVICE=cpu
QUERY_PREFIX="search_query: "
DOCUMENT_PREFIX="search_document: "
ENCODER_SERVICE_PORT=8260
```

**.env.deepvk:**
```bash
ENCODER_NAME=deepvk
HUGGING_FACE_MODEL_NAME=deepvk/USER2-base
DEVICE=mps
QUERY_PREFIX="search_query: "
DOCUMENT_PREFIX="search_document: "
ENCODER_SERVICE_PORT=8261
```

### Офлайн-режим

Сервис поддерживает работу в полностью офлайн-режиме при наличии локально скачанных моделей.

Модели хранятся в директории `MODEL_PATH` со следующей структурой:
```
models/sentence-transformers/
├── ai-forever/
│   └── FRIDA/
│       ├── config.json
│       ├── modules.json
│       ├── model.safetensors
│       └── tokenizer_config.json
└── deepvk/
    └── USER2-base/
        ├── config.json
        ├── modules.json
        ├── model.safetensors
        └── tokenizer_config.json
```

При запуске сервис:
1. Проверяет наличие модели по пути `{MODEL_PATH}/{HUGGING_FACE_MODEL_NAME}`
2. Если модель найдена локально — загружает её оттуда
3. Если модель не найдена — скачивает с Hugging Face Hub

Для принудительного офлайн-режима можно установить переменные окружения (в `main.py`):
```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

### Мониторинг

#### Health check
```bash
curl http://localhost:8260/health
```

Ответ включает метрики:
- `queue_size` — текущий размер очереди
- `active_requests` — количество ожидающих запросов
- `encoder_loaded` — загружена ли модель
- `status` — "healthy" или "degraded"
- `max_timeout` — максимальный таймаут сервиса

### Безопасность

- **Аутентификация по API-ключу** — все рабочие эндпоинты защищены статическим ключом
- **Очередь с ограничением** — защита от перегрузки (maxsize=1000)
- **Валидация входных данных** — Pydantic модели с проверкой границ
- **Изоляция контейнеров** — запуск под непривилегированным пользователем
- **Health checks** — для автоматического восстановления

### Клиентская библиотека (Encoder Client)

Для удобства работы с несколькими экземплярами Encoder Service предоставляется клиентская библиотека `EncoderClient`. Она полностью совместима с TEI-эндпоинтами.

#### Основные возможности

- Поддержка нескольких энкодеров одновременно
- Параллельные запросы к разным энкодерам
- Ленивое получение метаданных (/info) при первом использовании
- Автоматическая подпись запросов (API-ключ в заголовке)
- Ретраи и таймауты на уровне HTTP-клиента

#### Пример использования

```python
from shared.encoder_client import encoder_client

# Кодирование текста всеми доступными энкодерами
embeddings = await encoder_client.encode_text(
    text="Пример текста",
    request_type="query"
)

# Результат: {"deepvk": [0.123, ...], "frida": [0.456, ...]}

# Пакетное кодирование
batch_embeddings = await encoder_client.encode_batch(
    texts=["текст1", "текст2"],
    request_type="document"
)

# Подсчет токенов
token_counts = await encoder_client.tokenize(
    text="Пример текста",
    use_encoders=["deepvk"]  # только указанные энкодеры
)

# Закрытие клиента
await encoder_client.close()
```

#### Конфигурация клиента

Клиент использует переменные окружения из конфига проекта:

```python
# Из конфига проекта
ENCODERS = {
    "deepvk": "http://encoder-deepvk:8261",
    "frida": "http://encoder-frida:8260",
    "frida_tei": "http://tei-frida:8080"  # можно добавлять TEI-инстансы
}

# Таймауты HTTP-клиента (не передаются в теле запроса)
ENCODER_CLIENT_HTTP_TIMEOUT = 30        # для одиночных запросов
ENCODER_CLIENT_HTTP_BATCH_TIMEOUT = 120 # для batch-запросов
```