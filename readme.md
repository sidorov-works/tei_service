# Encoder Service

TEI-совместимый сервис для векторизации текста с поддержкой моделей Sentence Transformers.

## Особенности

- **TEI-совместимость** — эндпоинты `/embed`, `/tokenize`, `/info`, `/health` соответствуют спецификации Hugging Face TEI
- **Потокобезопасность** — единственный воркер гарантирует корректную работу с не-thread-safe моделями
- **Асинхронная архитектура** — очереди задач сглаживают пиковую нагрузку
- **Мультиплатформенность** — поддержка CPU, NVIDIA GPU (CUDA), Apple Silicon (MPS)
- **Расширенная валидация** — защита от перегрузки на уровне эндпоинтов
- **Docker-ready** — готовые образы для Windows и Linux с GPU поддержкой

## Быстрый старт

### Локальный запуск (Mac с Metal)

```bash
# Клонирование
git clone <repository-url>
cd encoder-service

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.mps.txt

# Настройка окружения
cp .env.frida.example .env.frida
# Отредактируйте .env.frida:
# DEVICE=mps
# HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA

# Запуск через honcho (рекомендуется)
honcho start -f procfile -e .env.frida

# Или напрямую uvicorn
uvicorn encoder_service.main:app --port 8262 --workers 1
```

### Локальный запуск (Linux с CPU/GPU)

```bash
# Установка системных зависимостей
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Создание и активация venv
python3 -m venv venv
source venv/bin/activate

# Установка PyTorch (выберите нужную версию)
# Для CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Для CUDA 12.1:
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Установка остальных зависимостей
pip install -r requirements.cuda.txt

# Настройка .env
cp .env.example .env
# Укажите DEVICE=cuda или DEVICE=cpu

# Запуск
uvicorn encoder_service.main:app --port 8262 --workers 1
```

## Запуск в Docker

### Docker на Windows (NVIDIA GPU)

**Требования:**
- Docker Desktop с WSL2 backend
- NVIDIA Container Toolkit
- NVIDIA драйверы

```bash
# 1. Создайте необходимые папки
mkdir models logs

# 2. Настройте .env файлы
cp .env.example .env
cp .env.example .env.encoder1
cp .env.example .env.encoder2

# 3. Отредактируйте .env.encoder1 и .env.encoder2:
# HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
# MAX_MODEL_BATCH_SIZE=32
# DEVICE=cuda

# 4. Запустите сервисы (базовый образ соберется автоматически)
docker-compose up --build

# Проверка GPU
docker exec encoder_service_1 nvidia-smi
```

**Важные настройки для Windows:**
- В Docker Desktop → Settings → Resources → WSL Integration → включите интеграцию
- Убедитесь, что файлы имеют правильные окончания строк (LF для скриптов в контейнере)
- Используйте `git config --global core.autocrlf input` перед клонированием

### Docker на Linux (NVIDIA GPU)

```bash
# Установка Docker и NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Запуск сервисов
mkdir models logs
cp .env.example .env
cp .env.example .env.encoder1

# Редактируем .env.encoder1:
# DEVICE=cuda
# MAX_MODEL_BATCH_SIZE=32

# Сборка и запуск
docker-compose up --build

# Или для одного экземпляра:
docker run --gpus all \
  -p 8260:8260 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  --env-file .env.encoder1 \
  encoder-service:latest \
  uvicorn encoder_service.main:app --host 0.0.0.0 --port 8260
```

## Docker файлы

### dockerfile.base

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    libopenblas-dev \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements для CUDA (без torch)
COPY requirements.cuda.txt .

# Устанавливаем зависимости (без torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.cuda.txt

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
```

### dockerfile.service

```dockerfile
FROM encoder-base:latest

# Устанавливаем PyTorch с CUDA
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

COPY . .

RUN mkdir -p logs models && \
    useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8260}/health || exit 1
```

### docker-compose.yml

```yaml
services:
  encoder1:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8260:8260"
    volumes:
      - ./models:/app/models
      - ./logs/encoder1:/app/logs
    env_file:
      - .env
      - .env.encoder1
    environment:
      - PORT=8260
      - DOCKER_ENV=true
    container_name: encoder_service_1
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8260 --host 0.0.0.0 --workers 1
      --lifespan on --timeout-graceful-shutdown 15
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

  encoder2:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8261:8261"
    volumes:
      - ./models:/app/models
      - ./logs/encoder2:/app/logs
    env_file:
      - .env
      - .env.encoder2
    environment:
      - PORT=8261
      - DOCKER_ENV=true
    container_name: encoder_service_2
    restart: unless-stopped
    command: >
      uvicorn encoder_service.main:app
      --port 8261 --host 0.0.0.0 --workers 1
      --lifespan on --timeout-graceful-shutdown 15
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

## Файлы зависимостей

### requirements.mps.txt (для Mac)

```txt
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.12.0
certifi==2025.11.12
cffi==2.0.0
charset-normalizer==3.4.4
click==8.3.1
concurrent-log-handler==0.9.28
cryptography==46.0.5
Deprecated==1.3.1
dotenv==0.9.9
ecdsa==0.19.1
fastapi==0.128.0
filelock==3.20.1
fsspec==2025.12.0
h11==0.16.0
hf-xet==1.2.0
honcho==2.0.0
huggingface-hub==0.36.0
idna==3.11
Jinja2==3.1.6
joblib==1.5.3
limits==5.8.0
logger-utils @ git+https://github.com/sidorov-works/logger_utils@v0.1.7
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.0
packaging==25.0
portalocker==3.2.0
pyasn1==0.6.2
pycparser==3.0
pydantic==2.12.5
pydantic_core==2.41.5
pygelf==0.4.3
python-dotenv==1.2.1
python-jose==3.5.0
python-json-logger==4.0.0
PyYAML==6.0.3
regex==2025.11.3
requests==2.32.5
rsa==4.9.1
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.16.3
sentence-transformers==5.2.0
setproctitle==1.3.7
six==1.17.0
slowapi==0.1.9
starlette==0.50.0
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.1
torch==2.9.1
tqdm==4.67.1
transformers==4.57.3
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.6.2
uvicorn==0.40.0
uvloop==0.22.1
wrapt==2.1.1
```

### requirements.cuda.txt (для Docker)

```txt
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.12.0
certifi==2025.11.12
charset-normalizer==3.4.4
click==8.3.1
concurrent-log-handler==0.9.28
dotenv==0.9.9
fastapi==0.128.0
filelock==3.20.1
fsspec==2025.12.0
h11==0.16.0
hf-xet==1.2.0
huggingface-hub==0.36.0
idna==3.11
Jinja2==3.1.6
joblib==1.5.3
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.0
packaging==25.0
portalocker==3.2.0
pydantic==2.12.5
pydantic_core==2.41.5
pygelf==0.4.3
python-dotenv==1.2.1
python-jose==3.5.0
python-json-logger==4.0.0
PyYAML==6.0.3
regex==2025.11.3
requests==2.32.5
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.16.3
sentence-transformers==5.2.0
setproctitle==1.3.7
slowapi==0.1.9
starlette==0.50.0
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.1
tqdm==4.67.1
transformers==4.57.3
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.6.2
uvicorn==0.40.0
logger-utils @ git+https://github.com/sidorov-works/logger_utils@v0.1.7
```

## Procfile (для локального запуска)

```
# Запуск через honcho
encoder_service: uvicorn encoder_service.main:app --port ${ENCODER_SERVICE_PORT} --workers 1 --lifespan on --timeout-graceful-shutdown 10
```

## Эндпоинты

### `GET /info`

Информация о загруженной модели и ее доступных промптах.

**Response**
```json
{
  "model_id": "ai-forever/FRIDA",
  "max_input_length": 512,
  "max_client_batch_size": 128,
  "prompts": [
    {"name": "query", "text": "search_query: "},
    {"name": "document", "text": "search_document: "}
  ]
}
```

*Примечание:* Поле `prompts` заполняется только для моделей, поддерживающих промпты. Для моделей без поддержки промптов возвращается пустой массив `[]`.

### `GET /health`

Проверка здоровья сервиса.

- `200 OK` — сервис готов принимать запросы
- `503 Service Unavailable` — сервис не готов (модель не загружена, очереди переполнены)

### `POST /embed`

Получение эмбеддингов текста.

**Request**
```json
{
  "inputs": ["текст1", "текст2"],
  "prompt_name": null,
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
  "inputs": ["текст1", "текст2"],
  "add_special_tokens": true,
  "truncate": true
}
```

**Response** — всегда массив массивов объектов TokenInfo

```json
[
  [
    {
      "id": 2,
      "text": "▁Привет",
      "special": false,
      "start": 0,
      "stop": 6
    },
    {
      "id": 6,
      "text": ",",
      "special": false,
      "start": 6,
      "stop": 7
    }
  ]
]
```

## Конфигурация

### Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|------------|----------|---------------------|
| **Основные** | | |
| `HUGGING_FACE_MODEL_NAME` | Имя модели на Hugging Face Hub | `ai-forever/FRIDA` |
| `DEVICE` | Устройство (`cpu`, `cuda`, `mps`) | `cpu` |
| `ENCODER_NAME` | Имя экземпляра сервиса | `frida` |
| `ENCODER_SERVICE_PORT` | Порт для запуска (используется в Procfile) | `8262` |
| **Размеры батчей** | | |
| `MAX_MODEL_BATCH_SIZE` | Размер батча для модели (GPU memory) | `32` |
| `MAX_SERVICE_BATCH_SIZE` | Макс. размер батча для эндпоинта | `128` |
| **Ограничения текста** | | |
| `MAX_TEXT_LENGTH` | Макс. длина одного текста (символы) | `10000` |
| `MAX_TOTAL_BATCH_LENGTH` | Макс. суммарная длина текстов в батче | `500000` |
| **Таймауты** | | |
| `EMBED_TIMEOUT` | Таймаут операции /embed (сек) | `30.0` |
| `TOKENIZE_TIMEOUT` | Таймаут операции /tokenize (сек) | `5.0` |
| **Очереди** | | |
| `INPUT_QUEUE_MAXSIZE` | Макс. размер входящей очереди | `1000` |
| `OUTPUT_QUEUE_MAXSIZE` | Макс. размер исходящей очереди | `1000` |
| `HEALTH_QUEUE_THRESHOLD` | Порог заполнения очереди для health check | `0.9` |
| **Токенизация** | | |
| `TOKENIZE_MODE` | Режим токенизации (`full` / `lite`) | `full` |
| **Rate limiting** | | |
| `RATE_LIMIT_INFO` | Лимит запросов к /info | `500/minute` |
| `RATE_LIMIT_HEALTH` | Лимит запросов к /health | `500/minute` |
| `RATE_LIMIT_EMBED` | Лимит запросов к /embed | `200/minute` |
| `RATE_LIMIT_COUNT_TOKENS` | Лимит запросов к /tokenize | `600/minute` |

### Пример .env файла (общий)

```bash
# .env
INTERNAL_API_SECRET=your_secret_key_here
LOG_PATH=logs
LOGGING_LEVEL=INFO
DOCKER_ENV=false

EMBED_TIMEOUT=30.0
TOKENIZE_TIMEOUT=5.0

INPUT_QUEUE_MAXSIZE=1000
OUTPUT_QUEUE_MAXSIZE=1000

MAX_SERVICE_BATCH_SIZE=128
MAX_TEXT_LENGTH=10000
MAX_TOTAL_BATCH_LENGTH=500000

TOKENIZE_MODE=lite

RATE_LIMIT_INFO=500/minute
RATE_LIMIT_HEALTH=500/minute
RATE_LIMIT_EMBED=200/minute
RATE_LIMIT_COUNT_TOKENS=600/minute
```

### Пример .env.frida (для конкретной модели)

```bash
# .env.frida
ENCODER_NAME=frida
ENCODER_SERVICE_PORT=8262
DEVICE=mps
HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
MAX_MODEL_BATCH_SIZE=32
```

### Пример .env.encoder1 (для Docker)

```bash
# .env.encoder1
ENCODER_NAME=frida
HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
DEVICE=cuda
MAX_MODEL_BATCH_SIZE=32
```

## Аутентификация

Для защиты внутренних эндпоинтов используется Bearer token:

```bash
curl -X POST http://localhost:8262/embed \
  -H "Authorization: Bearer your_secret_key_here" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "test"}'
```

Для отключения аутентификации не задавайте `INTERNAL_API_SECRET`.

## Мониторинг и логи

- Логи сохраняются в `logs/encoder_service/app.log`
- Уровень логирования настраивается через `LOGGING_LEVEL`
- Health check эндпоинт доступен для систем мониторинга

## Требования к системе

### Минимальные:
- **CPU**: 2 cores, 4GB RAM
- **GPU**: 4GB VRAM для моделей ~500MB

### Рекомендуемые:
- **CPU**: 4 cores, 8GB RAM
- **GPU**: 8GB+ VRAM для больших батчей

### Проверка GPU:
```bash
# Linux
nvidia-smi

# Docker
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Python
python -c "import torch; print(torch.cuda.is_available())"
```

## Устранение неполадок

### Проблемы с MPS (Mac)
```bash
# Убедитесь, что PyTorch версии 2.0+
pip install torch>=2.0.0
# Проверка
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Проблемы с CUDA (Linux/Windows)
```bash
# Проверка версии CUDA
nvcc --version
# Проверка PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

### Docker проблемы
```bash
# Очистка кэша
docker system prune -a

# Проверка логов
docker logs encoder_service_1

# Проверка GPU в контейнере
docker exec encoder_service_1 nvidia-smi
```

### Проблемы с памятью
- Уменьшите `MAX_MODEL_BATCH_SIZE`
- Установите `MAX_SERVICE_BATCH_SIZE` меньше
- Увеличьте `shm_size` в docker-compose

## Структура проекта

```
encoder-service/
├── encoder_service/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── worker.py            # ModelWorker
│   └── dispatcher.py        # ResultDispatcher
├── shared/
│   ├── __init__.py
│   ├── config.py            # Конфигурация
│   ├── auth_service.py      # Аутентификация
│   └── tei_models.py        # Pydantic модели
├── models/                   # Кэш моделей
├── logs/                     # Логи
├── requirements.mps.txt      # Зависимости для Mac
├── requirements.cuda.txt     # Зависимости для Docker
├── dockerfile.base           # Базовый Docker образ
├── dockerfile.service        # Docker образ сервиса
├── docker-compose.yml        # Docker Compose
├── procfile                  # Для запуска через honcho
├── .env.example              # Пример общей конфигурации
├── .env.frida.example        # Пример конфигурации для FRIDA
└── README.md                 # Этот файл
```

## Лицензия

MIT