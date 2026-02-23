## Encoder Service

Сервис векторизации текста, предоставляющий HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Содержание
- [Архитектура](#архитектура)
- [API Эндпоинты](#api-эндпоинты)
- [Модели данных](#модели-данных)
- [Потокобезопасность](#потокобезопасность)
- [Установка и запуск](#установка-и-запуск)
- [Запуск в Docker](#запуск-в-docker)
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
    "url": "http://localhost:8001",
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
| `url` | str | Базовый URL для доступа к сервису | "http://encoder-rag:8260" |
| `model_info` | EncoderModelInfo | Информация о модели | см. выше |
| `status` | str | Статус сервиса | "operational" |

```python
class EncoderInfo(BaseModel):
    name: str
    url: str
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
ENCODER_SERVICE_URL=http://localhost:8001
ENCODE_TIMEOUT=30.0
ENCODE_BATCH_TIMEOUT=60.0
```

#### Запуск сервиса
```bash
uvicorn encoder_service.main:app --host 0.0.0.0 --port 8001 --workers 1
```

### Запуск в Docker

#### Dockerfile
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

### Конфигурация

Каждый экземпляр энкодера запускается со своим `.env` файлом:

**Пример .env для GPU:**
```env
ENCODER_NAME=rag
DEVICE=cuda
INTERNAL_API_SECRET=your-secret-key
ENCODER_SERVICE_URL=http://encoder_service:8260
ENCODE_TIMEOUT=30.0
ENCODE_BATCH_TIMEOUT=60.0
EMBEDDING_MODEL={"model": "deepvk/USER2-base", "vector_size": 768, "max_seq_length": 8192}
```

**Пример .env для CPU:**
```env
ENCODER_NAME=classifier
DEVICE=cpu
INTERNAL_API_SECRET=your-secret-key
ENCODER_SERVICE_URL=http://localhost:8001
ENCODE_TIMEOUT=30.0
ENCODE_BATCH_TIMEOUT=60.0
EMBEDDING_MODEL={"model": "ai-forever/FRIDA", "vector_size": 768, "max_seq_length": 512}
```

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