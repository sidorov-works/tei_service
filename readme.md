
# Encoder Service

Сервис векторизации текста, предоставляющий HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Содержание
- [Архитектура](#архитектура)
- [API Эндпоинты](#api-эндпоинты)
- [Модели данных](#модели-данных)
- [Управление таймаутами](#управление-таймаутами)
- [Лимиты и валидация](#лимиты-и-валидация)
- [Очередь и многопоточность](#очередь-и-многопоточность)
- [Аутентификация и подпись запросов](#аутентификация-и-подпись-запросов)
  - [Выбор метода аутентификации](#выбор-метода-аутентификации)
  - [JWT-аутентификация (режим по умолчанию)](#jwt-аутентификация-режим-по-умолчанию)
  - [Аутентификация по секретному заголовку](#аутентификация-по-секретному-заголовку)
- [Установка и запуск](#установка-и-запуск)
  - [Локальный запуск](#локальный-запуск)
  - [Запуск с honcho (Procfile)](#запуск-с-honcho-procfile)
  - [Запуск в Docker](#запуск-в-docker)
  - [Запуск нескольких экземпляров](#запуск-нескольких-экземпляров)
- [Конфигурация](#конфигурация)
  - [Переменные окружения](#переменные-окружения)
  - [Примеры .env файлов](#примеры-env-файлов)
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

- **Гибкие таймауты** — клиент может указать желаемое время ожидания (`timeout` в теле запроса), сервис уважает этот таймаут (в пределах своего жесткого потолка).

- **JWT аутентификация** — все рабочие эндпоинты защищены. Поддерживается два режима: JWT-токены с коротким сроком жизни (по умолчанию 30 секунд) или статический секретный заголовок для разработки.

- **Локальное хранение моделей** — модели скачиваются один раз и сохраняются локально в `models/sentence-transformers/`.

- **Защита от перегрузок** — очередь запросов с максимальным размером (1000 задач) и валидация размера входящих данных. При заполнении очереди > 900 новые запросы получают 503 Service Unavailable.

### API Эндпоинты

**Важно:** Все POST эндпоинты требуют аутентификации (JWT или секретный заголовок) через заголовок `Authorization: Bearer <token>`.

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/info` | GET | Информация об энкодере и его модели (не требует аутентификации) |
| `/encode` | POST | Кодирование одного текста |
| `/encode_batch` | POST | Пакетное кодирование нескольких текстов |
| `/count_tokens` | POST | Подсчет токенов в одном тексте |
| `/count_tokens_batch` | POST | Подсчет токенов для нескольких текстов |
| `/vector_size` | GET | Размерность вектора модели (не требует аутентификации) |
| `/max_length` | GET | Максимальная длина текста в токенах (не требует аутентификации) |
| `/health` | GET | Проверка здоровья сервиса (не требует аутентификации) |

#### Информация об энкодере (`/info`)

```bash
curl http://localhost:8260/info
```

```json
{
    "name": "deepvk",
    "model_info": {
        "name": "deepvk/USER2-base",
        "vector_size": 768,
        "max_seq_length": 8192,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: "
    },
    "status": "operational"
}
```

#### Кодирование текста (`/encode`) с опциональным таймаутом

```bash
curl -X POST http://localhost:8260/encode \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{
    "text": "Проблема с доступом к почте",
    "request_type": "query",
    "timeout": 30
  }'
```

```json
{
    "embedding": [0.123, -0.456, ...],
    "dimension": 768,
    "service_available": true
}
```

#### Пакетное кодирование (`/encode_batch`)

```bash
curl -X POST http://localhost:8260/encode_batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{
    "texts": ["текст1", "текст2"],
    "request_type": "document",
    "timeout": 60
  }'
```

```json
{
    "embeddings": [[0.123, -0.456, ...], [0.789, -0.321, ...]],
    "count": 2,
    "service_available": true
}
```

#### Подсчет токенов (`/count_tokens`)

```bash
curl -X POST http://localhost:8260/count_tokens \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{
    "text": "Пример текста для подсчета токенов",
    "timeout": 15
  }'
```

```json
{
    "tokens_count": 8,
    "service_available": true
}
```

#### Пакетный подсчет токенов (`/count_tokens_batch`)

```bash
curl -X POST http://localhost:8260/count_tokens_batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{
    "texts": ["текст1", "текст2"],
    "timeout": 30
  }'
```

```json
{
    "tokens_counts": [5, 8],
    "count": 2,
    "service_available": true
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

#### EncodeRequest

```python
{
    "text": "string",           # обязательное, не пустое, ≤ MAX_TEXT_LENGTH
    "request_type": "query",     # опционально, "query" или "document"
    "timeout": 30.0              # опционально, секунд
}
```

#### BatchEncodeRequest

```python
{
    "texts": ["string1", "string2"],  # обязательное, 1-256 текстов
    "request_type": "query",           # опционально
    "timeout": 60.0                    # опционально
}
```

#### BatchTokenCountRequest

```python
{
    "texts": ["string1", "string2"],  # обязательное, 1-256 текстов
    "timeout": 30.0                    # опционально
}
```

Валидация:
- Количество текстов ≤ `MAX_BATCH_SIZE` (256)
- Каждый текст не пустой и ≤ `MAX_TEXT_LENGTH` (10000 символов)
- Суммарная длина ≤ `MAX_TOTAL_BATCH_LENGTH` (500000 символов)

### Управление таймаутами

Сервис реализует гибкую систему таймаутов:

1. **Клиентский таймаут** — клиент может передать в запросе поле `timeout`, указав, сколько секунд он готов ждать ответа.

2. **Таймауты сервиса по умолчанию** — если клиент не указал таймаут, используются значения из конфига:
   - `ENCODER_BASE_TIMEOUT` (15с) для `/encode`, `/count_tokens`
   - `ENCODER_BATCH_TIMEOUT` (60с) для `/encode_batch`, `/count_tokens_batch`

3. **Жесткий потолок** — `MAX_SERVICE_TIMEOUT` (120с) — сервис никогда не ждет дольше этого значения, даже если клиент запросил больше.

**Логика определения времени ожидания:**
```python
if client_timeout:
    effective_timeout = min(client_timeout, MAX_SERVICE_TIMEOUT)
else:
    effective_timeout = min(default_timeout, MAX_SERVICE_TIMEOUT)
```

### Очередь и многопоточность

Сервис использует асинхронную очередь для обработки запросов:

- **input_queue**: максимальный размер 1000 задач
- При заполнении очереди > 900 новые запросы получают 503 Service Unavailable
- **Один воркер** последовательно обрабатывает задачи, имея эксклюзивный доступ к модели
- **Диспетчер результатов** направляет ответы от воркера к ожидающим Future
- Для избежания блокировки event loop, операции с Future (`set_result`, `set_exception`) выполняются в отдельных потоках через `ThreadPoolExecutor`

### Аутентификация и подпись запросов

#### Выбор метода аутентификации

Encoder Service поддерживает два метода аутентификации, которые переключаются простым изменением импорта в `main.py`:

```python
# Вариант 1: JWT-токены (рекомендуется для production)
from shared.auth_service import require_jwt_auth as require_auth

# Вариант 2: Статический секретный заголовок (для разработки и тестирования)
# from shared.auth_service import require_header_secret as require_auth
```

**JWT-аутентификация**:
- Более безопасна благодаря короткоживущим токенам (30 секунд)
- Защищает от replay-атак
- Требует синхронизации времени между сервисами

**Аутентификация по секретному заголовку**:
- Проще в реализации и отладке
- Подходит для разработки и тестирования
- Менее безопасна (статический ключ)
- Использует тот же `INTERNAL_API_SECRET`, но передает его напрямую в заголовке

#### JWT-аутентификация (режим по умолчанию)

Токен должен быть подписан с использованием HS256 и содержать следующие поля:

```json
{
    "iss": "client-service-name",     // кто выпустил токен
    "iat": 1700000000,                 // время выпуска (Unix timestamp)
    "exp": 1700000030,                  // время истечения (iat + 30 сек)
    "service": "client-service-name"    // имя сервиса-клиента
}
```

**Важно:**
- `exp` должно быть не более чем через 30 секунд после `iat` (значение по умолчанию)
- Часы клиента и сервера должны быть синхронизированы (допустимое расхождение - несколько секунд)

##### Генерация токена (примеры на разных языках)

**Python**
```python
import time
import jwt

SECRET_KEY = "your-32-char-secret-key-minimum"  # должен совпадать с INTERNAL_API_SECRET

def create_token(service_name: str = "my-service") -> str:
    current_time = int(time.time())
    payload = {
        "iss": service_name,
        "iat": current_time,
        "exp": current_time + 30,
        "service": service_name
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
```

**JavaScript/Node.js**
```javascript
const jwt = require('jsonwebtoken');
const SECRET_KEY = 'your-32-char-secret-key-minimum';

function createToken(serviceName = 'my-service') {
    const currentTime = Math.floor(Date.now() / 1000);
    const payload = {
        iss: serviceName,
        iat: currentTime,
        exp: currentTime + 30,
        service: serviceName
    };
    return jwt.sign(payload, SECRET_KEY, { algorithm: 'HS256' });
}
```

##### Возможные ошибки JWT

| Код | Ответ | Причина |
|-----|-------|---------|
| 403 | `{"detail": "Missing authentication token"}` | Отсутствует заголовок Authorization |
| 403 | `{"detail": "Invalid token: Signature verification failed"}` | Неправильный секретный ключ |
| 403 | `{"detail": "Invalid token: Signature has expired"}` | Токен просрочен |

#### Аутентификация по секретному заголовку

При использовании этого режима клиент просто передает статический секретный ключ в заголовке:

```bash
curl -X POST http://localhost:8260/encode \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-32-char-secret-key-minimum" \
  -d '{"text": "пример текста"}'
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
| `INTERNAL_API_SECRET` | Секретный ключ для аутентификации | **Нет значения** |
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
| `RATE_LIMIT_ENCODE` | Лимит для /encode | "500/minute" |
| `RATE_LIMIT_ENCODE_BATCH` | Лимит для /encode_batch | "150/minute" |
| `RATE_LIMIT_COUNT_TOKENS` | Лимит для /count_tokens | "600/minute" |
| `RATE_LIMIT_COUNT_TOKENS_BATCH` | Лимит для /count_tokens_batch | "200/minute" |
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

- **Гибкая аутентификация** — выбор между JWT и секретным заголовком
- **Короткое время жизни токена** (в режиме JWT) — 30 секунд защищает от replay-атак
- **Очередь с ограничением** — защита от перегрузки (maxsize=1000)
- **Валидация входных данных** — Pydantic модели с проверкой границ
- **Изоляция контейнеров** — запуск под непривилегированным пользователем
- **Health checks** — для автоматического восстановления

### Клиентская библиотека (Encoder Client)

Для удобства работы с несколькими экземплярами Encoder Service предоставляется клиентская библиотека `EncoderClient`. Она не входит в состав самого сервиса, но является рекомендуемым способом взаимодействия с ним.

#### Основные возможности

- Поддержка нескольких энкодеров одновременно
- Параллельные запросы к разным энкодерам
- Ленивое получение метаданных (/info) при первом использовании
- Автоматическая подпись запросов (JWT или секретный заголовок)
- Ретраи и таймауты

#### Пример использования

```python
from shared.encoder_client import encoder_client

# Кодирование текста всеми доступными энкодерами
embeddings = await encoder_client.encode_text(
    text="Пример текста",
    request_type="query",
    timeout=30.0  # клиентский таймаут
)

# Результат: {"deepvk": [0.123, ...], "frida": [0.456, ...]}

# Пакетное кодирование
batch_embeddings = await encoder_client.encode_batch(
    texts=["текст1", "текст2"],
    request_type="document",
    timeout=60.0
)

# Подсчет токенов
token_counts = await encoder_client.count_tokens(
    text="Пример текста",
    use_encoders=["deepvk"]  # только указанные энкодеры
)

# Закрытие клиента
await encoder_client.close()
```

#### Конфигурация клиента

Клиент использует те же переменные окружения, что и другие сервисы проекта:

```python
# Из конфига проекта
ENCODERS = {
    "deepvk": "http://encoder-deepvk:8261",
    "frida": "http://encoder-frida:8260"
}
ENCODER_CLIENT_HTTP_TIMEOUT = 30  # для одиночных запросов
ENCODER_CLIENT_HTTP_BATCH_TIMEOUT = 120  # для batch-запросов
```