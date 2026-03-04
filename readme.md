## Encoder Service - Актуализированная документация

```markdown
# Encoder Service

Сервис векторизации текста, предоставляющий HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Содержание
- [Архитектура](#архитектура)
- [API Эндпоинты](#api-эндпоинты)
- [Модели данных](#модели-данных)
- [Управление таймаутами](#управление-таймаутами)
- [Лимиты и Rate Limiting](#лимиты-и-rate-limiting)
- [Аутентификация (JWT)](#аутентификация-jwt)
- [Подпись запросов (для клиентов)](#подпись-запросов-jwt-аутентификация)
- [Установка и запуск](#установка-и-запуск)
- [Запуск в Docker](#запуск-в-docker)
- [Конфигурация](#конфигурация)
- [Мониторинг](#мониторинг)
- [Производительность](#производительность)
- [Безопасность](#безопасность)

### Архитектура

Сервис построен с учетом следующих ключевых требований:

- **Thread-safe работа с моделями** — SentenceTransformer модели не являются потокобезопасными. В сервисе реализована архитектура "очередь + воркер":
  - FastAPI эндпоинты только принимают запросы и ставят их в очередь
  - Один выделенный воркер последовательно обрабатывает задачи, имея эксклюзивный доступ к модели
  - Диспетчер результатов направляет ответы ожидающим запросам через `asyncio.Future`

- **Батчевая обработка** — поддержка пакетного кодирования нескольких текстов за один запрос для оптимальной производительности.

- **Гибкие таймауты** — клиент может указать желаемое время ожидания, сервис уважает этот таймаут (в пределах своего жесткого потолка).

- **JWT аутентификация** — все рабочие эндпоинты защищены JWT токенами с коротким сроком жизни (30 секунд).

- **Локальное хранение моделей** — модели скачиваются один раз и сохраняются локально.

- **Защита от перегрузок** — многоуровневая система: очередь запросов, валидация размера, rate limiting по IP.

### API Эндпоинты

**Важно:** Все POST эндпоинты (кроме `/health` и `/info`) требуют JWT аутентификацию через заголовок `Authorization: Bearer <token>`.

| Эндпоинт | Метод | Описание | Аутентификация | Rate Limit |
|----------|-------|----------|----------------|------------|
| `/info` | GET | Информация об энкодере и его модели | Нет | 500/min |
| `/encode` | POST | Кодирование одного текста | **JWT** | 200/min |
| `/encode_batch` | POST | Пакетное кодирование нескольких текстов | **JWT** | 60/min |
| `/count_tokens` | POST | Подсчет токенов в одном тексте | **JWT** | 200/min |
| `/count_tokens_batch` | POST | Подсчет токенов для нескольких текстов | **JWT** | 60/min |
| `/vector_size` | GET | Размерность вектора модели | Нет | 500/min |
| `/max_length` | GET | Максимальная длина текста в токенах | Нет | 500/min |
| `/health` | GET | Проверка здоровья сервиса | Нет | 500/min |

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

#### Пакетный подсчет токенов (`/count_tokens_batch`)

```bash
curl -X POST http://localhost:8260/count_tokens_batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{
    "texts": ["короткий текст", "очень длинный текст"],
    "timeout": 20
  }'
```

```json
{
    "tokens_counts": [3, 15],
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

#### EncoderModelInfo

| Поле | Тип | Описание |
|------|-----|----------|
| `name` | str | Название модели на Hugging Face Hub |
| `vector_size` | int | Размерность выходного вектора |
| `max_seq_length` | int | Максимальная длина текста в токенах |
| `query_prefix` | str | Префикс для поисковых запросов |
| `document_prefix` | str | Префикс для документов |

#### EncoderInfo

| Поле | Тип | Описание |
|------|-----|----------|
| `name` | str | Уникальное имя энкодера |
| `model_info` | EncoderModelInfo | Информация о модели |
| `status` | str | Статус сервиса |

#### EncodeRequest

```python
{
    "text": "string",           # обязательное
    "request_type": "query",     # "query" или "document", опционально
    "timeout": 30.0              # опционально, секунд
}
```

Валидация:
- Текст не может быть пустым
- Длина текста ≤ `MAX_TEXT_LENGTH`

#### BatchEncodeRequest

```python
{
    "texts": ["string1", "string2"],  # обязательное
    "request_type": "query",           # опционально
    "timeout": 60.0                     # опционально
}
```

Валидация:
- Батч не может быть пустым
- Количество текстов ≤ `MAX_BATCH_SIZE`
- Каждый текст не пустой и ≤ `MAX_TEXT_LENGTH`
- Суммарная длина ≤ `MAX_TOTAL_BATCH_LENGTH`

#### BatchTokenCountRequest

```python
{
    "texts": ["string1", "string2"],  # обязательное
    "timeout": 20.0                     # опционально
}
```

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

**Важно:** Клиент должен настроить HTTP таймауты своего HTTP-клиента так, чтобы они были **не меньше** максимального таймаута, который он готов ждать. Иначе HTTP-слой оборвет соединение раньше, чем сервис вернет ответ.

### Лимиты и Rate Limiting

#### 1. Очередь запросов
- Максимальный размер очереди: 1000 задач
- При заполнении 900+ новые запросы получают 503 Service Unavailable

#### 2. Валидация размера запросов

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `MAX_BATCH_SIZE` | 256 | Макс. текстов в батче |
| `MAX_TEXT_LENGTH` | 10000 | Макс. длина текста (символов) |
| `MAX_TOTAL_BATCH_LENGTH` | 100000 | Макс. суммарная длина |

#### 3. Rate Limiting

| Эндпоинт | Лимит |
|----------|-------|
| `/info`, `/health`, `/vector_size`, `/max_length` | 500/min |
| `/encode`, `/count_tokens` | 200/min |
| `/encode_batch`, `/count_tokens_batch` | 60/min |

### Аутентификация (JWT)

- **Алгоритм**: HS256
- **Срок жизни токена**: 30 секунд
- **Передача**: заголовок `Authorization: Bearer <token>`

#### Формат токена

```json
{
    "iss": "service-name",        // обязательное
    "iat": 1700000000,            // время выпуска
    "exp": 1700000030,            // истекает через 30 сек
    "service": "service-name"      // имя сервиса
}
```

### Конфигурация

#### Переменные окружения сервиса

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `ENCODER_NAME` | Уникальное имя экземпляра | "frida" |
| `INTERNAL_API_SECRET` | Секретный ключ для JWT | **Обязательный** |
| `DEVICE` | Устройство для вычислений | "cpu" |
| `HUGGING_FACE_MODEL_NAME` | Название модели | "ai-forever/FRIDA" |
| `QUERY_PREFIX` | Префикс для запросов | "search_query: " |
| `DOCUMENT_PREFIX` | Префикс для документов | "search_document: " |
| `ENCODER_BASE_TIMEOUT` | Таймаут для простых запросов (сек) | 15 |
| `ENCODER_BATCH_TIMEOUT` | Таймаут для batch запросов (сек) | 60 |
| `MAX_SERVICE_TIMEOUT` | Жесткий потолок таймаута (сек) | 120 |
| `MAX_BATCH_SIZE` | Макс. текстов в батче | 256 |
| `MAX_TEXT_LENGTH` | Макс. длина текста | 10000 |
| `MAX_TOTAL_BATCH_LENGTH` | Макс. суммарная длина | 100000 |
| `PORT` | Порт сервиса | 8260 |

#### Пример `.env` файла

```bash
# Обязательные параметры
INTERNAL_API_SECRET=your-32-char-secret-key-minimum

# Модель
ENCODER_NAME=deepvk
HUGGING_FACE_MODEL_NAME=deepvk/USER2-base
DEVICE=cpu

# Таймауты
ENCODER_BASE_TIMEOUT=15
ENCODER_BATCH_TIMEOUT=60
MAX_SERVICE_TIMEOUT=120

# Лимиты
MAX_BATCH_SIZE=256
MAX_TEXT_LENGTH=10000
```

### Мониторинг

#### Health check
```bash
curl http://localhost:8260/health
```

Ответ включает метрики:
- `queue_size` — текущий размер очереди
- `active_requests` — количество ожидающих запросов
- `max_timeout` — максимальный таймаут сервиса

### Запуск

#### Локальный запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск (всегда с 1 воркером!)
uvicorn encoder_service.main:app --host 0.0.0.0 --port 8260 --workers 1
```

#### Запуск в Docker

```bash
# Сборка образа
docker build -t encoder-service .

# Запуск контейнера
docker run -d \
  --name encoder \
  -p 8260:8260 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  encoder-service
```

### Производительность

- **Очередь + воркер** — предсказуемая нагрузка на модель
- **Батчевая обработка** — до 256 текстов за один запрос
- **Асинхронные эндпоинты** — не блокируют потоки FastAPI

**Метрики для оценки:**
- Размер очереди (в health check)
- Время обработки задач (в логах воркера)
- Количество активных запросов

### Безопасность

- **JWT аутентификация** — все рабочие эндпоинты
- **Короткое время жизни токена** — 30 секунд
- **Очередь с ограничением** — защита от перегрузки
- **Rate limiting** — защита от DDoS по IP
- **Валидация входных данных** — Pydantic модели
- **Изоляция контейнеров** — запуск под непривилегированным пользователем
```