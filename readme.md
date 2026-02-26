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
HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
VECTOR_SIZE=1536
MAX_SEQ_LENGTH=512
QUERY_PREFIX=search_query:
DOCUMENT_PREFIX=search_document:
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
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - DOCKER_ENV=false  # для использования файлового логгера
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
├── encoder_service/           # код сервиса
├── shared/                    # общие модули
├── requirements.windows.txt   # зависимости
│
├── dockerfile.base            # для базового образа
├── dockerfile.service         # для экземпляров
│
├── .env.encoder1              # настройки экземпляра 1
├── .env.encoder2              # настройки экземпляра 2
│
├── models/                     # общая папка для моделей
│   └── sentence-transformers/  # сюда скачаются модели
│
├── logs/                       # папка для логов
│   ├── encoder1/               # логи экземпляра 1
│   └── encoder2/               # логи экземпляра 2
│
└── docker-compose.yml          # запуск всех экземпляров
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

**.env.encoder1** (первая модель):
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

**.env.encoder2** (вторая модель):
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
      # Общая папка для моделей
      - ./models:/app/models
      # Отдельная папка для логов
      - ./logs/encoder1:/app/logs
    env_file:
      - .env.encoder1
    environment:
      - PORT=8260
      - DOCKER_ENV=false  # для использования файлового логгера
    container_name: encoder_service_1
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

  # Второй экземпляр
  encoder2:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8261:8261"
    volumes:
      # Общая папка для моделей
      - ./models:/app/models
      # Отдельная папка для логов
      - ./logs/encoder2:/app/logs
    env_file:
      - .env.encoder2
    environment:
      - PORT=8261
      - DOCKER_ENV=false
    container_name: encoder_service_2
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8261 --host 0.0.0.0 --workers 1
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

#### Команды для работы с несколькими экземплярами

**Создание необходимых папок:**
```bash
mkdir -p logs\encoder1 logs\encoder2 models\sentence-transformers
```

**Сборка базового образа (один раз):**
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

**Просмотр логов:**
```bash
# Через docker-compose
docker-compose logs -f encoder1
docker-compose logs -f encoder2

# Или через файлы на хосте
type logs\encoder1\app.log
type logs\encoder2\app.log
```

**Остановка:**
```bash
docker-compose down
```

### Конфигурация

Конфигурация сервиса задается через файл `.env` и код в `shared/config.py`:

```python
# shared/config.py
from dotenv import load_dotenv
import os
from pathlib import Path
from shared.encoder_models import EncoderModelInfo

load_dotenv(override=True) 

class DefaultConfig:
    ENCODER_NAME = os.getenv("ENCODER_NAME", "frida")
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")
    EXC_INFO = False
    LOG_PATH = Path("logs")
    MODEL_PATH = Path("models") / "sentence-transformers"
    DEVICE = os.getenv("DEVICE", "cpu")
    
    EMBEDDING_MODEL = EncoderModelInfo(
        name=os.getenv("HUGGING_FACE_MODEL_NAME", "ai-forever/FRIDA"),
        vector_size=int(os.getenv("VECTOR_SIZE", "1536")),
        max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512")),
        query_prefix=os.getenv("QUERY_PREFIX", "search_query: "),
        document_prefix=os.getenv("DOCUMENT_PREFIX", "search_document: ")
    )

config = DefaultConfig()
```

Параметры окружения (`.env`):
| Параметр | Описание | Пример |
|----------|----------|--------|
| `ENCODER_NAME` | Уникальное имя экземпляра | `encoder1` |
| `INTERNAL_API_SECRET` | Секретный ключ | `secret-key-1` |
| `DEVICE` | Устройство для вычислений | `cuda` / `cpu` / `mps` |
| `HUGGING_FACE_MODEL_NAME` | Название модели | `deepvk/USER2-base` |
| `VECTOR_SIZE` | Размер вектора | `768` |
| `MAX_SEQ_LENGTH` | Макс. длина в токенах | `8192` |
| `QUERY_PREFIX` | Префикс для запросов | `search_query:` |
| `DOCUMENT_PREFIX` | Префикс для документов | `search_document:` |
| `PORT` | Порт сервиса | `8260` |

### Логирование

Сервис использует гибкую систему логирования из `shared.logger`:

```python
# shared/logger.py
import logging
import os
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pathlib import Path
from shared.config import config

def setup_logger(name: Optional[str] = None, ...) -> logging.Logger:
    """Логгер для разработки (пишет в файл)"""
    # использует ConcurrentRotatingFileHandler
    # с ротацией и сжатием
    pass

def setup_graylog_logger(...) -> logging.Logger:
    """Логгер для продакшена (отправляет в Graylog)"""
    pass

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Автовыбор логгера по DOCKER_ENV"""
    if os.environ.get('DOCKER_ENV') == 'true':
        return setup_graylog_logger(...)
    return setup_logger(name=name)

logger = get_logger("my_app")
```

**Особенности:**
- Автоматический выбор режима (`DOCKER_ENV=true/false`)
- Потокобезопасная запись в файл
- Ротация логов (10 MB, 5 бэкапов)
- Сжатие старых логов (use_gzip=True)
- Включение имени процесса в каждый лог
- В контейнерах с `DOCKER_ENV=false` логи пишутся в `/app/logs/app.log` и пробрасываются на хост через volumes

**Пример логов:**
```
2024-01-01 12:00:00 | INFO     | encoder1     | Starting encoder instance: encoder1
2024-01-01 12:00:01 | INFO     | encoder1     | Модель успешно загружена: deepvk/USER2-base
2024-01-01 12:00:02 | INFO     | encoder1     | Processing batch: 5 texts
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

#### Логи
```bash
# Просмотр логов конкретного экземпляра
tail -f logs/encoder1/app.log
```

### Производительность

- **Батчевая обработка** — модель эффективно обрабатывает до 32 текстов за один запрос
- **Кэширование моделей** — модели сохраняются в `models/sentence-transformers/` и переиспользуются
- **Общий кэш для нескольких экземпляров** — при запуске нескольких сервисов модели скачиваются один раз
- **Очистка памяти** — при завершении работы освобождаются ресурсы GPU/MPS
- **Таймауты** — настраиваемые таймауты для разных типов запросов

### Безопасность

- **Внутренняя аутентификация** — все эндпоинты, кроме `/health` и `/info`, защищены заголовком `X-Internal-Secret`
- **Очистка входных данных** — удаление нестандартных символов из текста
- **Валидация через Pydantic** — проверка структуры запросов
- **Информация о модели** — эндпоинт `/info` открыт, так как не содержит чувствительных данных
- **Изоляция контейнеров** — каждый экземпляр работает под непривилегированным пользователем `appuser`

### Разработка и отладка

#### Локальный запуск с отладкой
```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск с автоматической перезагрузкой при изменении кода
uvicorn encoder_service.main:app --reload --port 8001
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

#### Полезные команды для отладки

**Проверка загрузки моделей:**
```bash
# Посмотреть, какие модели скачаны
ls -la models/sentence-transformers/
```

**Мониторинг использования GPU:**
```bash
# В отдельном терминале
watch -n 1 nvidia-smi
```

**Очистка кэша моделей (если нужно перекачать):**
```bash
# Удалить конкретную модель
rm -rf models/sentence-transformers/deepvk%2FUSER2-base/
```

#### Типичные проблемы и решения

| Проблема | Решение |
|----------|---------|
| Модель не скачивается | Проверьте интернет-соединение, переменную `HUGGING_FACE_MODEL_NAME` |
| Out of memory на GPU | Уменьшите batch size, используйте `count: 1` в deploy.resources |
| Конфликт портов | Измените PORT в .env файлах |
| Нет логов в папке | Проверьте `DOCKER_ENV=false`, права на папку logs |
