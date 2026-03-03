# Encoder Service

Сервис векторизации текста, предоставляющий HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Содержание
- [Архитектура](#архитектура)
- [API Эндпоинты](#api-эндпоинты)
- [Модели данных](#модели-данных)
- [Потокобезопасность](#потокобезопасность)
- [Лимиты и Rate Limiting](#лимиты-и-rate-limiting)
- [Установка и запуск](#установка-и-запуск)
- [Запуск в Docker](#запуск-в-docker)
- [Запуск нескольких экземпляров](#запуск-нескольких-экземпляров)
- [Конфигурация](#конфигурация)
- [Логирование](#логирование)
- [Обработка ошибок](#обработка-ошибок)
- [Мониторинг](#мониторинг)
- [Производительность](#производительность)
- [Безопасность](#безопасность)
- [Разработка и отладка](#разработка-и-отладка)

### Архитектура

Сервис построен с учетом следующих ключевых требований:

- **Thread-safe работа с моделями** — SentenceTransformer модели не являются потокобезопасными, поэтому все эндпоинты, работающие с моделью, сделаны синхронными. FastAPI автоматически ставит запросы в очередь, гарантируя последовательную обработку.
- **Батчевая обработка** — поддержка пакетного кодирования нескольких текстов за один запрос для оптимальной производительности.
- **Локальное хранение моделей** — модели скачиваются один раз и сохраняются локально для последующих запусков.
- **Изоляция через HTTP** — сервис работает как отдельный процесс.
- **Самоидентификация** — каждый экземпляр энкодера имеет уникальное имя и предоставляет информацию о своей модели через отдельный эндпоинт.
- **Автоопределение параметров** — размер вектора и максимальная длина текста определяются автоматически из загруженной модели.
- **Защита от перегрузок** — многоуровневая система лимитов: валидация размера запросов + rate limiting по IP/сервисам.

### API Эндпоинты

Все эндпоинты, работающие с моделью (кроме `/health` и `/info`), требуют заголовок `X-Internal-Secret` для аутентификации.

| Эндпоинт | Метод | Описание | Аутентификация | Rate Limit |
|----------|-------|----------|----------------|------------|
| `/info` | GET | Информация об энкодере и его модели | Нет | 500/min |
| `/encode` | POST | Кодирование одного текста | Да | 200/min |
| `/encode_batch` | POST | Пакетное кодирование нескольких текстов | Да | 60/min |
| `/count_tokens` | POST | Подсчет токенов в одном тексте | Да | 200/min |
| `/count_tokens_batch` | POST | Подсчет токенов для нескольких текстов | Да | 60/min |
| `/vector_size` | GET | Размерность вектора модели | Да | 500/min |
| `/max_length` | GET | Максимальная длина текста в токенах | Да | 500/min |
| `/health` | GET | Проверка здоровья сервиса | Нет | 500/min |

#### Информация об энкодере (`/info`)

Возвращает структурированную информацию, необходимую клиентам для работы:

```bash
curl http://localhost:8001/info
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

#### Кодирование текста (`/encode`)

```bash
curl -X POST http://localhost:8001/encode \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: your-secret-key" \
  -d '{"text": "Проблема с доступом к почте", "request_type": "query"}'
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
curl -X POST http://localhost:8001/encode_batch \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: your-secret-key" \
  -d '{"texts": ["текст1", "текст2"], "request_type": "document"}'
```

```json
{
    "embeddings": [[0.123, -0.456, ...], [0.789, -0.321, ...]],
    "count": 2,
    "service_available": true
}
```

#### Пакетный подсчет токенов (`/count_tokens_batch`) 🚀

**Новый эндпоинт для эффективной обработки множества текстов.**

Позволяет подсчитать количество токенов для нескольких текстов за один запрос. Это критически важно для производительности при индексации больших документов: например, для 121 чанка с 2 энкодерами заменяет 242 отдельных запроса на всего 2.

```bash
curl -X POST http://localhost:8001/count_tokens_batch \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: your-secret-key" \
  -d '{"texts": ["короткий текст", "очень длинный текст с множеством слов и предложений"]}'
```

**Пример ответа:**
```json
{
    "tokens_counts": [3, 15],
    "count": 2,
    "service_available": true
}
```

#### Проверка здоровья (`/health`)

```bash
curl http://localhost:8001/health
```

```json
{
    "status": "healthy",
    "encoder_loaded": true,
    "service_available": true
}
```

### Модели данных

#### shared/encoder_models.py

Pydantic модели для описания энкодеров и их свойств с валидацией входных данных.

##### EncoderModelInfo

Информация о модели, используемой в энкодере.

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `name` | str | Название модели на Hugging Face Hub | "deepvk/USER2-base" |
| `vector_size` | int | Размерность выходного вектора | 768 |
| `max_seq_length` | int | Максимальная длина текста в токенах | 8192 |
| `query_prefix` | str | Префикс для поисковых запросов | "search_query: " |
| `document_prefix` | str | Префикс для документов при индексации | "search_document: " |

##### EncoderInfo

Полное описание экземпляра Encoder Service.

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `name` | str | Уникальное имя энкодера | "deepvk" |
| `model_info` | EncoderModelInfo | Информация о модели | см. выше |
| `status` | str | Статус сервиса | "operational" |

#### Модели запросов с валидацией

##### EncodeRequest

```python
class EncodeRequest(BaseModel):
    text: str
    request_type: Optional[str] = "query"  # "query" или "document"
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > config.MAX_TEXT_LENGTH:
            raise ValueError(f'Text length exceeds {config.MAX_TEXT_LENGTH} chars')
        return v
```

##### BatchEncodeRequest

```python
class BatchEncodeRequest(BaseModel):
    texts: List[str]
    request_type: Optional[str] = "query"

    @field_validator('texts')
    @classmethod
    def validate(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Batch cannot be empty')
        if len(v) > config.MAX_BATCH_SIZE:
            raise ValueError(f'Batch size exceeds {config.MAX_BATCH_SIZE}')
        
        total_len = 0
        for i, text in enumerate(v):
            text_len = len(text.strip())
            if text_len == 0:
                raise ValueError(f'Text at index {i} is empty')
            if text_len > config.MAX_TEXT_LENGTH:
                raise ValueError(f'Text at index {i} too long')
            total_len += text_len
            if total_len > config.MAX_TOTAL_BATCH_LENGTH:
                raise ValueError(f'Total batch length exceeds limit')
        
        return v
```

##### BatchTokenCountRequest

```python
class BatchTokenCountRequest(BaseModel):
    texts: List[str]
    # Та же валидация, что и у BatchEncodeRequest
```

##### BatchTokenCountResponse

```python
class BatchTokenCountResponse(BaseModel):
    tokens_counts: List[int]
    service_available: bool
    
    @computed_field
    @property
    def count(self) -> int:
        return len(self.tokens_counts)
```

### Потокобезопасность

**Критически важно:** SentenceTransformer модели не являются потокобезопасными. Параллельные вызовы `encoder.encode()` из разных потоков могут привести к segmentation fault.

Решение:
- Все эндпоинты, использующие модель, объявлены как **синхронные** (`def`, не `async def`)
- FastAPI для синхронных эндпоинтов автоматически ставит запросы в очередь
- Запросы обрабатываются последовательно, гарантируя безопасность

**Batch-эндпоинты** следуют тому же принципу: внутри одного запроса тексты обрабатываются последовательно, а сам запрос стоит в очереди FastAPI.

### Лимиты и Rate Limiting

Сервис использует двухуровневую систему защиты от перегрузок:

#### 1. Валидация размера запросов (Pydantic)

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `MAX_BATCH_SIZE` | 256 | Максимальное количество текстов в одном батче |
| `MAX_TEXT_LENGTH` | 10000 | Максимальная длина одного текста (символов) |
| `MAX_TOTAL_BATCH_LENGTH` | 100000 | Максимальная суммарная длина всех текстов в батче |

Эти лимиты предотвращают отправку заведомо невозможных для обработки запросов (например, 10 тысяч текстов по 1 млн символов).

#### 2. Rate Limiting (защита от DDoS)

Ограничивает частоту запросов от одного клиента. Используется in-memory хранилище (не требует Redis).

| Эндпоинт | Лимит | Назначение |
|----------|-------|------------|
| `/info`, `/health`, `/vector_size`, `/max_length` | 500/min | Легкие эндпоинты, можно чаще |
| `/encode`, `/count_tokens` | 200/min | Средняя нагрузка |
| `/encode_batch`, `/count_tokens_batch` | 60/min | Тяжелые операции |

**Как определяется клиент:**
- Если есть заголовок `X-Internal-Secret` — по первым 8 символам секрета (идентификация сервиса)
- Если нет — по IP адресу (с поддержкой `X-Forwarded-For`)

**При превышении лимита:**
- Возвращается HTTP 429 Too Many Requests
- В ответе есть заголовки: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- В логах фиксируется факт превышения (без спама)

### Установка и запуск

#### Предварительные требования
- Python 3.9+
- CUDA (опционально, для GPU) или MPS (для Mac M1/M2)

#### Установка зависимостей
```bash
pip install -r requirements.txt
```

#### Настройка конфигурации
Создайте файл `.env`:
```env
# Обязательные параметры
ENCODER_NAME=deepvk
INTERNAL_API_SECRET=your-secret-key
HUGGING_FACE_MODEL_NAME=deepvk/USER2-base

# Опционально
DEVICE=cpu                                # или "mps", "cuda"
QUERY_PREFIX=search_query:                # Префикс для запросов
DOCUMENT_PREFIX=search_document:          # Префикс для документов
PORT=8260                                 # Порт сервиса

# Лимиты (опционально, есть значения по умолчанию)
MAX_BATCH_SIZE=256
MAX_TEXT_LENGTH=10000
MAX_TOTAL_BATCH_LENGTH=100000

# Rate limiting (опционально)
RATE_LIMIT_INFO=500/minute
RATE_LIMIT_ENCODE=200/minute
RATE_LIMIT_ENCODE_BATCH=60/minute
RATE_LIMIT_COUNT_TOKENS=200/minute
RATE_LIMIT_COUNT_TOKENS_BATCH=60/minute
```

#### Запуск сервиса
```bash
uvicorn encoder_service.main:app --host 0.0.0.0 --port 8260 --workers 1
```

### Запуск в Docker

#### Базовый Dockerfile
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    curl libopenblas-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p logs models
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8260/health || exit 1
```

#### docker-compose.yml
```yaml
services:
  encoder_service:
    build: .
    ports:
      - "8260:8260"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - DOCKER_ENV=false
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8260 --host 0.0.0.0 --workers 1
      --loop uvloop --lifespan on --timeout-graceful-shutdown 15
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
```

### Конфигурация

Полный список параметров конфигурации:

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `ENCODER_NAME` | Уникальное имя экземпляра | "frida" |
| `INTERNAL_API_SECRET` | Секретный ключ | None (обязательный) |
| `DEVICE` | Устройство для вычислений | "cpu" |
| `HUGGING_FACE_MODEL_NAME` | Название модели | "ai-forever/FRIDA" |
| `QUERY_PREFIX` | Префикс для запросов | "search_query: " |
| `DOCUMENT_PREFIX` | Префикс для документов | "search_document: " |
| `PORT` | Порт сервиса | 8260 |
| `MAX_BATCH_SIZE` | Макс. текстов в батче | 256 |
| `MAX_TEXT_LENGTH` | Макс. длина текста | 10000 |
| `MAX_TOTAL_BATCH_LENGTH` | Макс. суммарная длина | 100000 |
| `RATE_LIMIT_INFO` | Лимит для /info и легких | "500/minute" |
| `RATE_LIMIT_ENCODE` | Лимит для /encode | "200/minute" |
| `RATE_LIMIT_ENCODE_BATCH` | Лимит для /encode_batch | "60/minute" |
| `RATE_LIMIT_COUNT_TOKENS` | Лимит для /count_tokens | "200/minute" |
| `RATE_LIMIT_COUNT_TOKENS_BATCH` | Лимит для /count_tokens_batch | "60/minute" |
| `DOCKER_ENV` | Режим окружения | false |
| `EXC_INFO` | Выводить stacktrace в лог | false |

### Обработка ошибок

Сервис возвращает ошибки в едином формате:

#### 400 Bad Request (ошибка валидации)
```json
{
    "error": "Validation error",
    "detail": "texts -> 0: Text cannot be empty",
    "service_available": true
}
```

#### 429 Too Many Requests (превышен rate limit)
```json
{
    "error": "Rate limit exceeded",
    "detail": "60 per minute",
    "service_available": true
}
```
В ответе также присутствуют заголовки:
- `X-RateLimit-Limit: 60`
- `X-RateLimit-Remaining: 0`
- `X-RateLimit-Reset: 1625097660`

#### 503 Service Unavailable (модель не загружена)
```json
{
    "error": "Encoder service not available",
    "service_available": false
}
```

### Мониторинг

#### Health check
```bash
curl http://localhost:8001/health
```

#### Информация об энкодере
```bash
curl http://localhost:8001/info
```

#### Проверка rate limit (текущее состояние)
```bash
# После запроса смотрим заголовки ответа
X-RateLimit-Limit: 200
X-RateLimit-Remaining: 199
X-RateLimit-Reset: 1625097660
```

### Производительность

- **Батчевая обработка** — модель эффективно обрабатывает до 256 текстов за один запрос
- **Пакетный подсчет токенов** — позволяет обрабатывать сотни текстов за 2-3 запроса вместо сотен отдельных
- **Кэширование моделей** — модели сохраняются на диске и переиспользуются
- **Общий кэш для нескольких экземпляров** — модели скачиваются один раз
- **Rate limiting** — защищает от случайных и намеренных перегрузок

**Пример эффективности с учетом rate limiting:**
- Для индексации документа со 121 чанком и 2 энкодерами:
  - Без batch: 242 запроса (превысит rate limit в 200/min)
  - С batch: 2 запроса (комфортно в лимитах)

### Безопасность

- **Внутренняя аутентификация** — заголовок `X-Internal-Secret` для всех рабочих эндпоинтов
- **Rate limiting** — защита от DDoS и случайных перегрузок
- **Валидация входных данных** — Pydantic модели с проверкой размеров
- **Очистка текста** — удаление нестандартных символов
- **Очистка эмбеддингов** — замена NaN на 0, проверка на Inf
- **Изоляция контейнеров** — запуск под непривилегированным пользователем

### Разработка и отладка

#### Локальный запуск
```bash
uvicorn encoder_service.main:app --reload --port 8001
```

#### Тестирование rate limiting
```bash
# Быстрые запросы для проверки лимитов
for i in {1..10}; do
  curl -X POST http://localhost:8260/encode \
    -H "X-Internal-Secret: test" \
    -H "Content-Type: application/json" \
    -d '{"text": "test"}' \
    -w "\n%{http_code}\n"
done
```

#### Полезные команды

**Проверка загрузки моделей:**
```bash
ls -la models/sentence-transformers/
```

**Мониторинг GPU:**
```bash
watch -n 1 nvidia-smi
```

**Проверка текущих лимитов:**
```bash
curl -I http://localhost:8260/info
# Смотрим заголовки X-RateLimit-*
```

#### Типичные проблемы и решения

| Проблема | Решение |
|----------|---------|
| Модель не скачивается | Проверьте интернет, `HUGGING_FACE_MODEL_NAME` |
| Out of memory на GPU | Уменьшите `MAX_BATCH_SIZE` в .env |
| Слишком много 429 ошибок | Увеличьте лимиты или оптимизируйте клиент (используйте batch) |
| Конфликт портов | Измените `PORT` в .env |
| Rate limit не работает | Проверьте, что во все эндпоинты добавлен декоратор `@limiter.limit` |
| Request validation error | Проверьте размер текста, не превышает ли лимиты |
| Нет логов в папке | Установите `DOCKER_ENV=false` |