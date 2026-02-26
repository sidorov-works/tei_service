## Encoder Service

Сервис векторизации текста, предоставляющий HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Содержание
- [Архитектура](#архитектура)
- [API Эндпоинты](#api-эндпоинты)
- [Модели данных](#модели-данных)
- [Потокобезопасность](#потокобезопасность)
- [Установка и запуск](#установка-и-запуск)
- [Запуск в Docker](#запуск-в-docker)
- [Запуск нескольких экземпляров](#запуск-нескольких-экземпляров)
- [Конфигурация](#конфигурация)
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

### API Эндпоинты

Все эндпоинты, работающие с моделью (кроме `/health` и `/info`), требуют заголовок `X-Internal-Secret` для аутентификации.

| Эндпоинт | Метод | Описание | Аутентификация |
|----------|-------|----------|----------------|
| `/info` | GET | Информация об энкодере и его модели | Нет |
| `/encode` | POST | Кодирование одного текста | Да |
| `/encode_batch` | POST | Пакетное кодирование нескольких текстов | Да |
| `/count_tokens` | POST | Подсчет токенов в тексте | Да |
| `/vector_size` | GET | Размерность вектора модели | Да |
| `/max_length` | GET | Максимальная длина текста в токенах | Да |
| `/health` | GET | Проверка здоровья сервиса | Нет |

#### Информация об энкодере (`/info`)

Возвращает структурированную информацию, необходимую клиентам для работы:

```bash
curl http://localhost:8001/info
```

```json
{
    "name": "rag",
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

Pydantic модели для описания энкодеров и их свойств.

##### EncoderModelInfo

Информация о модели, используемой в энкодере.

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `name` | str | Название модели на Hugging Face Hub | "deepvk/USER2-base" |
| `vector_size` | int | Размерность выходного вектора | 768 |
| `max_seq_length` | int | Максимальная длина текста в токенах | 8192 |
| `query_prefix` | str | Префикс для поисковых запросов | "search_query: " |
| `document_prefix` | str | Префикс для документов при индексации | "search_document: " |

```python
class EncoderModelInfo(BaseModel):
    name: str
    vector_size: int
    max_seq_length: int
    query_prefix: str = ""
    document_prefix: str = ""
    
    class Config:
        frozen = True
```

##### EncoderInfo

Полное описание экземпляра Encoder Service.

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `name` | str | Уникальное имя энкодера | "rag" |
| `model_info` | EncoderModelInfo | Информация о модели | см. выше |
| `status` | str | Статус сервиса | "operational" |

```python
class EncoderInfo(BaseModel):
    name: str
    model_info: EncoderModelInfo
    status: str = "operational"
    
    class Config:
        frozen = True
```

#### Модели запросов

##### EncodeRequest

```python
class EncodeRequest(BaseModel):
    text: str
    request_type: Optional[str] = "query"  # "query" или "document"
```

##### BatchEncodeRequest

```python
class BatchEncodeRequest(BaseModel):
    texts: List[str]
    request_type: Optional[str] = "query"  # "query" или "document"
```

### Потокобезопасность

**Критически важно:** SentenceTransformer модели не являются потокобезопасными. Параллельные вызовы `encoder.encode()` из разных потоков могут привести к segmentation fault.

Решение:
- Все эндпоинты, использующие модель, объявлены как **синхронные** (`def`, не `async def`)
- FastAPI для синхронных эндпоинтов автоматически ставит запросы в очередь
- Запросы обрабатываются последовательно, гарантируя безопасность

Пример работы очереди:
1. Первый запрос → FastAPI начинает обработку
2. Второй запрос → FastAPI ставит в очередь
3. Завершение обработки первого запроса → отправка ответа
4. Начало обработки второго запроса → отправка ответа

### Установка и запуск

#### Предварительные требования
- Python 3.9+
- CUDA (опционально, для GPU) или MPS (для Mac M1/M2)

#### Установка зависимостей
```bash
pip install -r requirements.txt
```

#### Скачивание модели
```bash
python -m shared.utils.download_models
```

#### Настройка конфигурации
Создайте файл `.env`:
```env
ENCODER_NAME=rag                      # Уникальное имя этого энкодера
INTERNAL_API_SECRET=your-secret-key
DEVICE=cpu                            # или "mps" для Mac, "cuda" для NVIDIA
```

#### Запуск сервиса
```bash
uvicorn encoder_service.main:app --host 0.0.0.0 --port 8001 --workers 1
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

#### docker-compose.yml для одного экземпляра
```yaml
services:
  encoder_service:
    build: .
    ports:
      - "8260:8260"
    volumes:
      - ./models:/app/models
    env_file:
      - .env
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

### Запуск нескольких экземпляров

Для ситуаций, когда нужно одновременно запустить несколько копий сервиса с разными моделями (например, для тестирования, поддержки разных языков или разделения нагрузки), используется двухуровневая система образов.

#### Как это работает

1. **Базовый слой** — один раз устанавливается всё тяжелое окружение (Python, пакеты, CUDA-библиотеки)
2. **Слой экземпляра** — легкая надстройка для каждой копии с кодом и настройками

```
┌─────────────────────┐      ┌─────────────────────┐
│  Экземпляр 1        │      │  Экземпляр 2        │
│  ┌─────────────────┐│      │  ┌─────────────────┐│
│  │ Код + .env1     ││      │  │ Код + .env2     ││
│  └─────────────────┘│      │  └─────────────────┘│
│  ┌─────────────────┐│      │  ┌─────────────────┐│
│  │   Базовый слой  ││      │  │   Базовый слой  ││
│  │  (общий для всех)││      │  │  (общий для всех)││
│  └─────────────────┘│      │  └─────────────────┘│
└─────────────────────┘      └─────────────────────┘
```

#### Структура папок для нескольких экземпляров

```
encoder-service/
│
├── dockerfile.base           # для базового образа
├── dockerfile.service        # для экземпляров
├── requirements.windows.txt  # общие зависимости
│
├── encoder_service/           # код сервиса
├── shared/                    # общие модули
│
├── .env.encoder1              # настройки экземпляра 1
├── .env.encoder2              # настройки экземпляра 2
│
├── models/
│   ├── encoder1/              # модель экземпляра 1
│   └── encoder2/              # модель экземпляра 2
│
├── logs/
│   ├── encoder1/              # логи экземпляра 1
│   └── encoder2/              # логи экземпляра 2
│
└── docker-compose.yml         # запуск всех экземпляров
```

#### Базовый образ (dockerfile.base)

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl libopenblas-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.windows.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.windows.txt

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
```

Сборка базового образа (делается один раз):
```bash
docker build -t encoder-base:latest -f dockerfile.base .
```

#### Образ для экземпляров (dockerfile.service)

```dockerfile
FROM encoder-base:latest

COPY . .

RUN mkdir -p logs models && \
    useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8260}/health || exit 1
```

#### Файлы .env для каждого экземпляра

**.env.encoder1** (модель deepvk/USER2-base):
```env
ENCODER_NAME=encoder1
INTERNAL_API_SECRET=secret-key-1
DEVICE=cuda
HUGGING_FACE_MODEL_NAME=deepvk/USER2-base
VECTOR_SIZE=768
MAX_SEQ_LENGTH=8192
QUERY_PREFIX=search_query:
DOCUMENT_PREFIX=search_document:
PORT=8260
```

**.env.encoder2** (модель ai-forever/FRIDA):
```env
ENCODER_NAME=encoder2
INTERNAL_API_SECRET=secret-key-2
DEVICE=cuda
HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
VECTOR_SIZE=1536
MAX_SEQ_LENGTH=512
QUERY_PREFIX=search_query:
DOCUMENT_PREFIX=search_document:
PORT=8261
```

#### Docker Compose для нескольких экземпляров

```yaml
services:
  # Первый экземпляр
  encoder1:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8260:8260"
    volumes:
      - ./models/encoder1:/app/models
      - ./logs/encoder1:/app/logs
    env_file:
      - .env.encoder1
    environment:
      - PORT=8260
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8260 --host 0.0.0.0 --workers 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Второй экземпляр
  encoder2:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8261:8261"
    volumes:
      - ./models/encoder2:/app/models
      - ./logs/encoder2:/app/logs
    env_file:
      - .env.encoder2
    environment:
      - PORT=8261
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8261 --host 0.0.0.0 --workers 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### Команды для работы с несколькими экземплярами

**Сборка базового образа:**
```bash
docker build -t encoder-base:latest -f dockerfile.base .
```

**Запуск всех экземпляров:**
```bash
docker-compose up -d
```

**Проверка работы:**
```bash
curl http://localhost:8260/info
curl http://localhost:8261/info
```

**Просмотр логов конкретного экземпляра:**
```bash
docker-compose logs -f encoder1
```

**Остановка всех:**
```bash
docker-compose down
```

#### Важные замечания при запуске нескольких экземпляров

1. **Память GPU** — если видеокарта одна, модели делят её память. Убедитесь, что обе модели помещаются одновременно.

2. **Отдельные папки** — всегда используйте разные папки для моделей (`models/encoder1`, `models/encoder2`), чтобы избежать конфликтов.

3. **Разные порты** — каждый экземпляр должен слушать свой уникальный порт.

4. **Первый запуск** — при первом запуске каждый экземпляр скачает свою модель (это может занять время и место).

5. **Ресурсы CPU/RAM** — учитывайте, что каждый экземпляр потребляет оперативную память под свою модель.

### Конфигурация

Конфигурация сервиса задается через файл `.env` и код в `shared/config.py`. Основные параметры:

```python
class DefaultConfig:

    ENCODER_NAME = os.getenv("ENCODER_NAME", "frida")
    
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")

    EXC_INFO = False # выводить ли в лог весь stacktrace

    LOG_PATH = Path("logs")
    MODEL_PATH = Path("models") / "sentence-transformers"

    DEVICE = os.getenv("DEVICE", "cpu")  # или "cpu"/"cuda"/"mps"

    # Описание и свойства эмбеддинговой модели
    EMBEDDING_MODEL = EncoderModelInfo(
        name=os.getenv("HUGGING_FACE_MODEL_NAME", "ai-forever/FRIDA"), 
        vector_size=int(os.getenv("VECTOR_SIZE", "1536")),
        max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512")),
        query_prefix=os.getenv("QUERY_PREFIX", "search_query: "),
        document_prefix=os.getenv("DOCUMENT_PREFIX", "search_document: ")
    )
```

Параметры окружения (`.env`):
- `ENCODER_NAME` - уникальное имя экземпляра энкодера
- `INTERNAL_API_SECRET` - секретный ключ для аутентификации
- `DEVICE` - устройство для вычислений (`cpu`, `cuda`, `mps`)
- `HUGGING_FACE_MODEL_NAME` - название модели на Hugging Face Hub
- `VECTOR_SIZE` - размер вектора модели
- `MAX_SEQ_LENGTH` - максимальная длина входного текста (в токенах)
- `QUERY_PREFIX` - префикс для поисковых запросов
- `DOCUMENT_PREFIX` - префикс для документов

Модели сохраняются в директорию `models/sentence-transformers/`.

### Обработка ошибок

Сервис использует многоуровневую систему обработки ошибок:

1. **На уровне API** — возврат информативных сообщений об ошибках
2. **На уровне модели** — очистка NaN значений в эмбеддингах

Пример ответа с ошибкой:
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

### Производительность

- **Батчевая обработка** — модель эффективно обрабатывает до 32 текстов за один запрос
- **Кэширование модели** — модель загружается в память один раз при старте
- **Очистка памяти** — при завершении работы освобождаются ресурсы GPU/MPS
- **Таймауты** — настраиваемые таймауты для разных типов запросов

### Безопасность

- **Внутренняя аутентификация** — все эндпоинты, кроме `/health` и `/info`, защищены заголовком `X-Internal-Secret`
- **Очистка входных данных** — удаление нестандартных символов из текста
- **Валидация через Pydantic** — проверка структуры запросов
- **Информация о модели** — эндпоинт `/info` открыт, так как не содержит чувствительных данных

### Разработка и отладка

#### Логирование
```python
logger.info(f"Starting encoder instance: {self.encoder_name}")
logger.info(f"Processing batch: {len(texts)} texts")
logger.error(f"Encode endpoint error: {e}", exc_info=True)
```

#### Очистка эмбеддингов
```python
def _clean_embedding(embedding: List[float]) -> Optional[List[float]]:
    cleaned_embedding = []
    for x in embedding:
        if math.isnan(x):
            cleaned_embedding.append(0)
        elif math.isinf(x):
            raise Exception("Inf value in embedding")
        else:
            cleaned_embedding.append(x)
    return cleaned_embedding
```