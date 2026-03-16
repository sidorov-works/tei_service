# Encoder Service

TEI-совместимый HTTP сервис для векторизации текста.

## Описание

Encoder Service предоставляет HTTP API для работы с эмбеддинговыми моделями, полностью совместимый с [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) от Hugging Face.

Сервис реализует ключевые эндпоинты TEI и может использоваться как прозрачная замена оригинальному TEI в задачах:
- Получение эмбеддингов текста (`/embed`)
- Токенизация текста (`/tokenize`)
- Получение информации о модели (`/info`)
- Проверка здоровья сервиса (`/health`)

## Архитектура

Сервис построен на асинхронной очереди задач:

```
[HTTP Request] → [FastAPI] → [Input Queue] → [Model Worker] → [Output Queue] → [Dispatcher] → [HTTP Response]
```

- **Model Worker** — единственный владелец модели SentenceTransformer, гарантирует потокобезопасность
- **ResultDispatcher** — связывает результаты из очереди с ожидающими Future
- **Очереди** — сглаживают пиковую нагрузку и защищают модель от перегрузки
- **Таймауты** — тройная защита: таймаут операции, защита от переполнения очереди, очистка зависших задач

## Эндпоинты

### `GET /info`

Информация о загруженной модели.

**Response**
```json
{
  "model_id": "deepvk/USER2-base",
  "max_input_length": 512,
  "max_client_batch_size": 128
}
```

### `GET /health`

Проверка здоровья сервиса.

- `200 OK` — сервис готов принимать запросы
- `503 Service Unavailable` — сервис не готов (модель не загружена, очереди переполнены)

### `POST /embed`

Получение эмбеддингов текста.

**Request**
```json
{
  "inputs": ["text1", "text2"],
  "prompt_name": "query",
  "normalize": false,
  "truncate": true,
  "truncation_direction": "right"
}
```

**Response** — всегда массив массивов float
```json
[
  [0.123, -0.456, 0.789, ...],
  [0.321, -0.654, 0.987, ...]
]
```

### `POST /tokenize`

Токенизация текста.

**Request**
```json
{
  "inputs": ["text1", "text2"],
  "add_special_tokens": true,
  "truncate": true
}
```

**Response** — всегда массив массивов объектов TokenInfo
```json
[
  [
    {
      "id": 101,
      "text": "[CLS]",
      "special": true,
      "start": 0,
      "stop": 0
    },
    {
      "id": 2054,
      "text": "what",
      "special": false,
      "start": 0,
      "stop": 4
    }
  ]
]
```

## Конфигурация

### Основные переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|---------------|
| `HUGGING_FACE_MODEL_NAME` | Имя модели на HF Hub | `ai-forever/FRIDA` |
| `DEVICE` | Устройство для инференса (`cpu`, `cuda`, `mps`) | `cpu` |
| `MAX_MODEL_BATCH_SIZE` | Размер батча для модели | `32` |
| `MAX_SERVICE_BATCH_SIZE` | Макс. размер батча для эндпоинта | `128` |
| `MAX_TEXT_LENGTH` | Макс. длина одного текста | `10000` |
| `EMBED_TIMEOUT` | Таймаут операции /embed (сек) | `30.0` |
| `TOKENIZE_TIMEOUT` | Таймаут операции /tokenize (сек) | `5.0` |
| `INPUT_QUEUE_MAXSIZE` | Макс. размер входящей очереди | `1000` |
| `INTERNAL_API_SECRET` | Секретный ключ для аутентификации | — |
| `TOKENIZE_MODE` | Режим токенизации (`full`/`lite`) | `full` |

### Префиксы для инструкций

Поддерживаются два типа префиксов, которые добавляются к тексту в зависимости от `prompt_name`:

- `QUERY_PREFIX` — для поисковых запросов (по умолчанию: `"search_query: "`)
- `DOCUMENT_PREFIX` — для индексируемых документов (по умолчанию: `"search_document: "`)

## Аутентификация

Сервис использует JWT-токены или секретный ключ в заголовке `Authorization: Bearer <token>`.

Для отключения аутентификации (например, в dev-окружении) можно не задавать `INTERNAL_API_SECRET`.

## Запуск

### Локально (с Honcho)

```bash
# Для модели FRIDA
honcho start -f procfile -e .env.frida

# Для модели DeepVK
honcho start -f procfile -e .env.deepvk
```

### В Docker (Windows с GPU)

```bash
# Подготовка папок
mkdir models logs\encoder1 logs\encoder2

# Сборка базового образа
docker build -f dockerfile.base -t encoder-base:latest .

# Сборка и запуск сервисов
docker-compose up --build
```

### Проверка GPU в контейнере

```bash
docker exec -it encoder_service_1 python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Мониторинг

- **Логи** — сохраняются в `logs/encoder_service/app.log`
- **Health check** — эндпоинт `/health` для проверки состояния
- **Rate limiting** — настраивается через переменные `RATE_LIMIT_*`

## Файловая структура проекта

```
encoder-service/
├── encoder_service/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── worker.py             # ModelWorker
│   └── dispatcher.py         # ResultDispatcher
├── shared/
│   ├── __init__.py
│   ├── config.py             # Конфигурация
│   ├── auth_service.py       # JWT аутентификация
│   └── tei_models.py         # Pydantic модели
├── models/                   # Кэш моделей (монтируемый volume)
├── logs/                     # Логи (монтируемый volume)
├── dockerfile.base           # Базовый образ с зависимостями
├── dockerfile.service        # Образ сервиса
├── docker-compose.yml        # Запуск нескольких экземпляров
├── requirements.windows.txt  # Зависимости для Windows
├── .env                      # Общие переменные
├── .env.encoder1             # Специфичные для первого экземпляра
└── .env.encoder2             # Специфичные для второго экземпляра
```

## Зависимости

- Python 3.11+
- PyTorch 2.3.1 (CUDA 12.1)
- Sentence-Transformers 5.2.0
- FastAPI 0.128.0
- Uvicorn 0.40.0
- logger-utils (внутренний пакет)

## Лицензия

MIT