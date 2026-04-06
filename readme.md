# TEI Service

TEI-совместимый сервис для векторизации текста и классификации. Поддерживает два режима работы:
- **Encoder** — эмбеддинги текста через Sentence Transformers
- **Classifier** — классификация текста через sequence classification модели

## Особенности

- **TEI-совместимость** — эндпоинты `/embed`, `/tokenize`, `/predict`, `/info`, `/health` соответствуют спецификации Hugging Face TEI
- **Два режима работы** — один сервис может работать как энкодер или как классификатор (выбирается через `SERVER_TYPE`)
- **Потокобезопасность** — единственный воркер гарантирует корректную работу с не-thread-safe моделями
- **Асинхронная архитектура** — очереди задач сглаживают пиковую нагрузку
- **Мультиплатформенность** — поддержка CPU, NVIDIA GPU (CUDA), Apple Silicon (MPS)
- **Расширенная валидация** — защита от перегрузки на уровне эндпоинтов
- **Docker-ready** — готовые образы для Linux и Windows с GPU поддержкой

## Быстрый старт

### Режим энкодера (Mac с Metal)

```bash
# Клонирование
git clone <repository-url>
cd tei-service

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.mps.txt

# Настройка окружения для энкодера
cp .env.frida.example .env.frida
# Отредактируйте .env.frida:
# SERVER_TYPE=encoder
# DEVICE=mps
# HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA

# Запуск через honcho (рекомендуется)
honcho start -f procfile -e .env.frida

# Или напрямую uvicorn
uvicorn main:app --port 8262 --workers 1
```

### Режим классификатора (Mac с Metal)

```bash
# Настройка окружения для классификатора
cp .env.pikabu.example .env.pikabu
# Отредактируйте .env.pikabu:
# SERVER_TYPE=classifier
# DEVICE=mps
# HUGGING_FACE_MODEL_NAME=sismetanin/rubert-toxic-pikabu-2ch

# Запуск
honcho start -f procfile -e .env.pikabu
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
# Укажите SERVER_TYPE=encoder или SERVER_TYPE=classifier
# Укажите DEVICE=cuda или DEVICE=cpu

# Запуск
uvicorn main:app --port 8262 --workers 1
```

## Запуск в Docker

### Требования

**Для Linux:**
- Docker Engine
- NVIDIA Container Toolkit (для GPU)
- NVIDIA драйверы (для GPU)

**Для Windows:**
- Docker Desktop с WSL2 backend
- NVIDIA Container Toolkit
- NVIDIA драйверы

### Подготовка

```bash
# 1. Создайте необходимые папки
mkdir models logs

# 2. Настройте .env файлы
cp .env.example .env
cp .env.encoder1.example .env.encoder1
cp .env.encoder2.example .env.encoder2

# 3. Отредактируйте .env.encoder1 и .env.encoder2:
#    SERVER_TYPE=encoder
#    HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
#    MAX_MODEL_BATCH_SIZE=32
#    DEVICE=cuda
```

### Сборка и запуск

```bash
# 4. Соберите базовый образ
docker build -f dockerfile.base -t tei-base:latest .

# 5. Запустите сервисы
docker-compose up --build

# 6. Проверка GPU (опционально)
docker exec tei_service_1 nvidia-smi
```

### Проверка работоспособности

```bash
# Health check
curl http://localhost:8260/health

# Информация о модели
curl http://localhost:8260/info

# Получение эмбеддинга (только в режиме энкодера)
curl -X POST http://localhost:8260/embed \
  -H "Authorization: Bearer your_secret_key_here" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Привет, мир!"}'

# Классификация текста (только в режиме классификатора)
curl -X POST http://localhost:8260/predict \
  -H "Authorization: Bearer your_secret_key_here" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Ты идиот!"}'
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

COPY requirements.cuda.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.cuda.txt

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
```

### dockerfile.service

```dockerfile
FROM tei-base:latest

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
  tei1:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8260:8260"
    volumes:
      - ./models:/app/models
      - ./logs/tei1:/app/logs
    env_file:
      - .env
      - .env.tei1
    environment:
      - PORT=8260
      - DOCKER_ENV=true
    container_name: tei_service_1
    restart: unless-stopped
    command: >
      uvicorn main:app
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

  tei2:
    build:
      context: .
      dockerfile: dockerfile.service
    ports:
      - "8261:8261"
    volumes:
      - ./models:/app/models
      - ./logs/tei2:/app/logs
    env_file:
      - .env
      - .env.tei2
    environment:
      - PORT=8261
      - DOCKER_ENV=true
    container_name: tei_service_2
    restart: unless-stopped
    command: >
      uvicorn main:app
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
logger-utils @ git+https://github.com/sidorov-works/logger_utils@v0.4.0
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
pygelf==4.0.3
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
pygelf==4.0.3
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
logger-utils @ git+https://github.com/sidorov-works/logger_utils@v0.4.0
```

## Procfile (для локального запуска)

```
tei: uvicorn main:app --port ${TEI_SERVICE_PORT} --workers 1 --lifespan on --timeout-graceful-shutdown 10
```

## Эндпоинты

### `GET /info`

Информация о загруженной модели. Доступен в обоих режимах.

**Response для энкодера:**
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

**Response для классификатора:**
```json
{
  "model_id": "sismetanin/rubert-toxic-pikabu-2ch",
  "max_input_length": 512,
  "max_client_batch_size": 128,
  "prompts": []
}
```

### `GET /health`

Проверка здоровья сервиса. Доступен в обоих режимах.

- `200 OK` — сервис готов принимать запросы
- `503 Service Unavailable` — сервис не готов (модель не загружена, очереди переполнены)

### `POST /embed` (только в режиме энкодера)

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

### `POST /tokenize` (только в режиме энкодера)

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

#### Режимы токенизации

Режим задается переменной окружения `TOKENIZE_MODE`:

- **`full`** (по умолчанию) — возвращает полную информацию о токенах (id, текст, флаг special, позиции в тексте)
- **`lite`** — возвращает только id токенов, остальные поля пустые (быстрее, меньше трафик)

### `POST /predict` (только в режиме классификатора)

Классификация текста.

**Request**
```json
{
  "inputs": ["текст1", "текст2"],
  "raw_scores": false,
  "truncate": true
}
```

**Response** — для одиночного текста возвращает массив объектов, для батча — массив массивов

```json
// Одиночный текст
[
  {"label": "toxic", "score": 0.98},
  {"label": "non-toxic", "score": 0.02}
]

// Батч
[
  [
    {"label": "toxic", "score": 0.98},
    {"label": "non-toxic", "score": 0.02}
  ],
  [
    {"label": "toxic", "score": 0.01},
    {"label": "non-toxic", "score": 0.99}
  ]
]
```

## Конфигурация

### Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|------------|----------|---------------------|
| **Режим работы** | | |
| `SERVER_TYPE` | Тип сервиса (`encoder` или `classifier`) | `encoder` |
| **Основные** | | |
| `HUGGING_FACE_MODEL_NAME` | Имя модели на Hugging Face Hub | `ai-forever/FRIDA` |
| `DEVICE` | Устройство (`cpu`, `cuda`, `mps`) | `cpu` |
| `SERVER_NAME` | Имя экземпляра сервиса | `tei` |
| `TEI_SERVICE_PORT` | Порт для запуска (используется в Procfile) | `8262` |
| **Размеры батчей** | | |
| `MAX_MODEL_BATCH_SIZE` | Размер батча для модели (GPU memory) | `32` |
| `MAX_SERVICE_BATCH_SIZE` | Макс. размер батча для эндпоинта | `128` |
| **Ограничения текста** | | |
| `MAX_TEXT_LENGTH` | Макс. длина одного текста (символы) | `10000` |
| `MAX_TOTAL_BATCH_LENGTH` | Макс. суммарная длина текстов в батче | `500000` |
| **Таймауты** | | |
| `EMBED_TIMEOUT` | Таймаут операции /embed (сек) | `30.0` |
| `TOKENIZE_TIMEOUT` | Таймаут операции /tokenize (сек) | `5.0` |
| `PREDICT_TIMEOUT` | Таймаут операции /predict (сек) | `30.0` |
| **Очереди** | | |
| `INPUT_QUEUE_MAXSIZE` | Макс. размер входящей очереди | `1000` |
| `OUTPUT_QUEUE_MAXSIZE` | Макс. размер исходящей очереди | `1000` |
| **Токенизация** | | |
| `TOKENIZE_MODE` | Режим токенизации (`full` / `lite`) | `full` |
| **Обработка NaN** | | |
| `EMBEDDING_CLEAN_NAN` | Заменять NaN/Inf в эмбеддингах | `true` |
| `EMBEDDING_NAN_REPLACEMENT` | Значение для замены NaN/Inf | `0.0` |
| `EMBEDDING_LOG_NAN` | Логировать факт замены NaN/Inf | `true` |
| **Rate limiting** | | |
| `RATE_LIMIT_INFO` | Лимит запросов к /info | `500/minute` |
| `RATE_LIMIT_HEALTH` | Лимит запросов к /health | `500/minute` |
| `RATE_LIMIT_EMBED` | Лимит запросов к /embed | `200/minute` |
| `RATE_LIMIT_TOKENIZE` | Лимит запросов к /tokenize | `600/minute` |
| `RATE_LIMIT_PREDICT` | Лимит запросов к /predict | `200/minute` |

### Пример .env файла (общий)

```bash
# .env

# --- Безопасность и аутентификация ---

INTERNAL_API_SECRET=your_extremely_safety_internal_api_secret
REQUIRE_AUTH=false

# --- Логирование ---

LOG_PATH=logs
LOGGING_LEVEL=INFO
LOG_FORMAT="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s"
DOCKER_ENV=false

# --- Ограничения длин очередей ---

OUTPUT_QUEUE_MAXSIZE=1000
INPUT_QUEUE_MAXSIZE=1000

# --- Обработка NaN/Inf в эмбеддингах ---
EMBEDDING_CLEAN_NAN=true           # Заменять ли NaN/Inf
EMBEDDING_NAN_REPLACEMENT=0.0      # На что заменять
EMBEDDING_LOG_NAN=true             # Логировать ли факт замены


# --- Лимиты на входящие запросы ---
# При превышении лимитов сервис должен вызвать ValidationError

# Максимально допустимое ДЛЯ СЕРВИСА кол-во текстов в батче. 
# Этот параметр не имеет отношения к модели. 
# Воркеры сервиса разбивают батч на подбатчи, подходящие для конкретной модели.
# А данный параметр является именно предварительным ограничением самого сервиса, 
# чтобы не произошел коллапс ресурсов.
MAX_SERVICE_BATCH_SIZE=128 

# Максимально допустимая длина одного текста в запросе. 
# Этот параметр не связан напрямую со свойством max_seq_len конкретной эмбеддинговой модели 
# Это просто подстраховка, чтобы не "забить" сервис заведомо огромными запросами
MAX_TEXT_LENGTH=10000

# Максимально допустимая суммарная длина текстов в батче
MAX_TOTAL_BATCH_LENGTH=500000


# --- Rate limiting и защита от Ddos ---

RATE_LIMIT_INFO=500/minute
RATE_LIMIT_HEALTH=500/minute
RATE_LIMIT_EMBED=200/minute              
RATE_LIMIT_TOKENIZE=600/minute
RATE_LIMIT_PREDICT=200/minute
```

### Пример .env.frida (для энкодера)

```bash
# .env.frida.example

SERVER_NAME=frida
SERVER_TYPE=encoder

SERVER_PORT=8260 # в коде не используется, нужно только для запуска с procfile

DEVICE=mps

HUGGING_FACE_MODEL_NAME=ai-forever/FRIDA
MAX_MODEL_BATCH_SIZE=32

# Режим применения эндпойнта /tokenize:
# Оригинальный TEI возвращает полную информацию о токенах. 
# Это не всегда и не всем требуется. Например, для основного применения - подсчета 
# длины текста в токенах - достаточно только определить длину списков с информацией о токенах, 
# а сама информация не нужна. Поэтому предусматриваем режим "lite", 
# в котором информация по токенам будет содержать типовые плейсхолдеры
TOKENIZE_MODE=lite # или full

# Тайм-ауты: максимальное время выполнения операции (от входа в эндпоинт до возврата результата)
EMBED_TIMEOUT=30.0
TOKENIZE_TIMEOUT=15.0
```

### Пример .env.pikabu (для классификатора)

```bash
# .env.pikabu.example

SERVER_NAME=pikabu
SERVER_TYPE=classifier

SERVER_PORT=8265 # в коде не используется, нужно только для запуска с procfile

DEVICE=mps

HUGGING_FACE_MODEL_NAME=sismetanin/rubert-toxic-pikabu-2ch
MAX_MODEL_BATCH_SIZE=32

# Тайм-ауты: максимальное время выполнения операции (от входа в эндпоинт до возврата результата)
PREDICT_TIMEOUT=30.0
```

### Пример .env.tei1 (для Docker)

```bash
# .env.tei1
SERVER_NAME=frida
SERVER_TYPE=encoder
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

- Логи сохраняются в `logs/tei_service/app.log`
- Уровень логирования настраивается через `LOGGING_LEVEL`
- Health check эндпоинт доступен для систем мониторинга

## Структура проекта

```
tei-service/
├── main.py                     # FastAPI приложение
├── dispatcher.py               # ResultDispatcher
├── workers/
│   ├── __init__.py
│   ├── base_worker.py          # BaseWorker (абстрактный)
│   ├── encoder_worker.py       # EncoderWorker
│   └── classifier_worker.py    # ClassifierWorker
├── shared/
│   ├── __init__.py
│   ├── config.py               # Конфигурация
│   ├── auth_service.py         # Аутентификация
│   ├── task.py                 # Task, TaskType, TaskResult
│   └── tei_models.py           # Pydantic модели
├── models/                     # Кэш моделей
│   ├── sentence-transformers/  # Для энкодера
│   └── transformers/           # Для классификатора
├── logs/                       # Логи
├── requirements.mps.txt        # Зависимости для Mac
├── requirements.cuda.txt       # Зависимости для Docker
├── dockerfile.base             # Базовый Docker образ
├── dockerfile.service          # Docker образ сервиса
├── docker-compose.yml          # Docker Compose
├── procfile                    # Для запуска через honcho
├── .env.example                # Пример общей конфигурации
├── .env.frida.example          # Пример конфигурации для FRIDA (энкодер)
├── .env.pikabu.example         # Пример конфигурации для pikabu (классификатор)
└── README.md                   # Этот файл
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
docker logs tei_service_1

# Проверка GPU в контейнере
docker exec tei_service_1 nvidia-smi
```

### Проблемы с памятью
- Уменьшите `MAX_MODEL_BATCH_SIZE`
- Установите `MAX_SERVICE_BATCH_SIZE` меньше
- Увеличьте `shm_size` в docker-compose

### Проблемы с окончаниями строк (Windows)
```bash
# Перед клонированием настройте Git
git config --global core.autocrlf input

# Или создайте .gitattributes в корне проекта:
echo "* text=auto" > .gitattributes
echo "*.py text eol=lf" >> .gitattributes
echo "dockerfile* text eol=lf" >> .gitattributes
```

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

## Лицензия

MIT