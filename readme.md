## Encoder Service – сервис векторизации текста

Сервис векторизации текста, предоставляющий HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Архитектура

Сервис построен с учетом следующих ключевых требований:
- **Thread-safe работа с моделями** — SentenceTransformer модели не являются потокобезопасными, поэтому все эндпоинты, работающие с моделью, сделаны синхронными. FastAPI автоматически ставит запросы в очередь, гарантируя последовательную обработку.
- **Батчевая обработка** — поддержка пакетного кодирования нескольких текстов за один запрос для оптимальной производительности.
- **Локальное хранение моделей** — модели скачиваются один раз и сохраняются локально для последующих запусков.
- **Изоляция через HTTP** — сервис работает как отдельный процесс, взаимодействие с другими компонентами через HTTP API с внутренней аутентификацией.

### Основные компоненты

#### FastAPI приложение (`encoder_service/main.py`)

Основной модуль реализует:
- Загрузку модели SentenceTransformer при старте
- Синхронные эндпоинты для кодирования (гарантия потокобезопасности)
- Очистку текста от нестандартных символов
- Валидацию входных данных через Pydantic модели
- Внутреннюю аутентификацию через секретный заголовок

#### Конфигурация (`shared/config.py`)

Централизованное управление настройками:
- Путь к модели и устройство (CPU/MPS/CUDA)
- Таймауты запросов
- Префиксы для query/document типов запросов
- Внутренний секрет для аутентификации

#### Клиент (`shared/encoder_client.py`)

Асинхронный клиент для взаимодействия с сервисом:
- Поддержка повторных попыток с экспоненциальной задержкой
- Таймауты на запросы
- Методы для всех эндпоинтов сервиса
- Глобальный инстанс для переиспользования

#### HTTP-клиент с ретраями (`shared/http_client.py`)

Универсальный клиент для надежных HTTP-запросов:
- Экспоненциальная задержка между попытками
- Настраиваемые таймауты
- Поддержка HTTP/2
- Лимиты соединений

### API Эндпоинты

Все эндпоинты (кроме `/health` и `/status`) требуют заголовок `X-Internal-Secret` для аутентификации.

| Эндпоинт | Метод | Описание | Важно |
|----------|-------|----------|-------|
| `/encode` | POST | Кодирование одного текста | Синхронный |
| `/encode_batch` | POST | Пакетное кодирование нескольких текстов | Синхронный, основной для воркеров |
| `/count_tokens` | POST | Подсчет токенов в тексте | Синхронный |
| `/vector_size` | GET | Размерность вектора модели | |
| `/max_length` | GET | Максимальная длина текста в токенах | |
| `/health` | GET | Проверка здоровья сервиса | Асинхронный |
| `/status` | GET | Детальный статус сервиса | Асинхронный |

### Модели данных

#### EncodeRequest
```python
{
    "text": "string",           # Текст для кодирования
    "request_type": "query"      # Опционально: "query" или "document"
}
```

#### BatchEncodeRequest
```python
{
    "texts": ["text1", "text2"], # Список текстов
    "request_type": "query"       # Опционально: "query" или "document"
}
```

### Потокобезопасность

**Критически важно:** SentenceTransformer модели не являются потокобезопасными. Параллельные вызовы `encoder.encode()` из разных потоков могут привести к segmentation fault.

Решение:
- Все эндпоинты, использующие модель, объявлены как **синхронные** (`def`, не `async def`)
- FastAPI для синхронных эндпоинтов автоматически ставит запросы в очередь
- Запросы обрабатываются последовательно, гарантируя безопасность

Пример работы очереди:
1. `w_classifier` отправляет запрос → FastAPI начинает обработку
2. `w_rag` отправляет запрос → FastAPI ставит в очередь
3. Завершение обработки запроса `w_classifier` → отправка ответа
4. Начало обработки запроса `w_rag` → отправка ответа

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
INTERNAL_API_SECRET=your-secret-key
DEVICE=cpu  # или "mps" для Mac, "cuda" для NVIDIA
ENCODER_SERVICE_URL=http://localhost:8001
ENCODE_TIMEOUT=30.0
ENCODE_BATCH_TIMEOUT=60.0
```

#### Запуск сервиса
```bash
uvicorn encoder_service.main:app --host 0.0.0.0 --port 8001
```

### Запуск в Docker

#### Запуск на компьютере с NVIDIA GPU (Windows в Docker Desktop)

##### Dockerfile для Windows

```dockerfile
# encoder_service/dockerfile.windows
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    curl libopenblas-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.windows.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.windows.txt
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

##### Docker Compose конфигурация

```yaml
# docker-compose.yml
services:
  encoder_service:
    build:
      context: .
      dockerfile: dockerfile.windows
    ports:
      - "8260:8260"
    volumes:
      - ./models:/app/models  # Кэш моделей на хосте
    env_file:
      - .env
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8260 --host 0.0.0.0 --workers 1
      --loop uvloop --lifespan on --timeout-graceful-shutdown 15
    init: true
    shm_size: '2gb'           # Shared memory для PyTorch
    mem_limit: 8g             # Лимит памяти
    cpus: '4.0'               # Лимит CPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all       # Использовать все доступные GPU
              capabilities: [gpu]
```

##### Важные нюансы запуска на Windows

1. **Требования к хосту**:
   - Установленный Docker Desktop с WSL2 backend
   - Установленный NVIDIA Container Toolkit для Windows
   - Драйверы NVIDIA с поддержкой CUDA

2. **Переменные окружения в `.env`**:
   ```env
   DEVICE=cuda  # Важно! Указываем использование GPU
   INTERNAL_API_SECRET=your-secret-key
   ENCODER_SERVICE_URL=http://encoder_service:8260
   ENCODE_TIMEOUT=30.0
   ENCODE_BATCH_TIMEOUT=60.0
   ```

3. **Проверка доступа к GPU**:
   ```bash
   # Проверка, что GPU доступен в контейнере
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   
   # Проверка логов сервиса
   docker logs encoder_service  # Должно быть "Использую устройство: cuda"
   ```

4. **Производительность**:
   - Благодаря `--workers 1` и синхронным эндпоинтам, очередь запросов управляется FastAPI
   - GPU используется последовательно, без конфликтов доступа
   - Батчевая обработка (`encode_batch`) оптимально утилизирует GPU

#### Локальный запуск на Mac (для разработки)

Для быстрого тестирования на MacBook с чипом M1/M2 используйте Overmind:

```
# Procfile
encoder_service: uvicorn encoder_service.main:app --port ${ENCODER_SERVICE_PORT} --workers 1 --lifespan on --timeout-graceful-shutdown 10
```

В `.env` для Mac:
```env
DEVICE=mps  # Использование Metal Performance Shaders
```

### Интеграция с другими сервисами

#### Пример использования клиента

```python
from shared.encoder_client import encoder_client

# Кодирование одного текста
embedding = await encoder_client.encode_text(
    text="Проблема с доступом к почте",
    request_type="query"  # или "document" для индексации
)

# Пакетное кодирование
embeddings = await encoder_client.encode_batch(
    texts=["текст1", "текст2"],
    request_type="document"
)

# Подсчет токенов
tokens = await encoder_client.get_tokens_count("текст для анализа")

# Получение информации о модели
vector_size = await encoder_client.get_vector_size()
max_length = await encoder_client.get_max_length()
model_name = await encoder_client.get_model_name()
```

#### Интеграция с RAG-подсистемой

Сервис активно используется в:
- **Индексации документов** (`document_service.py`) — для получения векторов чанков
- **RAG-поиске** (`rag_service.py`) — для кодирования поисковых запросов

Пример из `document_service.py`:
```python
# Получение размерности вектора для создания коллекции
self._vector_size = await self._encoder.get_vector_size()
```

Пример из `rag_service.py`:
```python
# Кодирование запроса для поиска
query_embedding = await self._encoder.encode_text(request.query)

# Пакетное кодирование нескольких запросов
query_embeddings = await self._encoder.encode_batch(queries)
```

### Обработка ошибок

Сервис использует многоуровневую систему обработки ошибок:

1. **На уровне HTTP-клиента** — повторные попытки с экспоненциальной задержкой
2. **На уровне API** — возврат информативных сообщений об ошибках
3. **На уровне модели** — очистка NaN значений в эмбеддингах

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
Ответ:
```json
{
    "status": "healthy",
    "encoder_loaded": true,
    "service_available": true
}
```

#### Статус сервиса
```bash
curl http://localhost:8001/status
```
Ответ:
```json
{
    "service": "Encoder Service",
    "status": "operational",
    "encoder_loaded": true,
    "model": "deepvk/USER2-base",
    "device": "cpu",
    "note": "All encode endpoints are synchronous for thread safety"
}
```

### Производительность

- **Батчевая обработка** — модель эффективно обрабатывает до 32 текстов за один запрос
- **Кэширование модели** — модель загружается в память один раз при старте
- **Очистка памяти** — при завершении работы освобождаются ресурсы GPU/MPS
- **Таймауты** — настраиваемые таймауты для разных типов запросов

### Безопасность

- **Внутренняя аутентификация** — все эндпоинты защищены заголовком `X-Internal-Secret`
- **Очистка входных данных** — удаление нестандартных символов из текста
- **Валидация через Pydantic** — проверка структуры запросов

### Разработка и отладка

#### Логирование
Сервис использует структурированное логирование с префиксом "ENCODER_SERVICE":
```python
logger.info(f"Processing batch: {len(texts)} texts")
logger.error(f"Encode endpoint error: {e}", exc_info=True)
```

#### Очистка эмбеддингов
Функция `_clean_embedding` заменяет NaN значения на 0 и выбрасывает исключение при Inf значениях:
```python
def _clean_embedding(embedding: List[float]) -> Optional[List[float]]:
    for x in embedding:
        if math.isnan(x):
            cleaned_embedding.append(0)
        elif math.isinf(x):
            raise Exception("Inf value in embedding")
```