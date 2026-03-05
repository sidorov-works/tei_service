Отлично! Вот полная актуальная документация, соответствующая текущему коду, с подробным описанием запуска:

##

# Encoder Service

Сервис векторизации текста, предоставляющий HTTP API для получения эмбеддингов через FastAPI. Использует модели на базе SentenceTransformers для преобразования текста в векторные представления.

### Содержание
- [Архитектура](#архитектура)
- [API Эндпоинты](#api-эндпоинты)
- [Модели данных](#модели-данных)
- [Управление таймаутами](#управление-таймаутами)
- [Лимиты и валидация](#лимиты-и-валидация)
- [Очередь и многопоточность](#очередь-и-многопоточность)
- [Аутентификация (JWT)](#аутентификация-jwt)
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

### Архитектура

Сервис построен с учетом следующих ключевых требований:

- **Thread-safe работа с моделями** — SentenceTransformer модели не являются потокобезопасными. В сервисе реализована архитектура "очередь + воркер":
  - FastAPI эндпоинты принимают запросы и ставят их в очередь `input_queue`
  - Один выделенный воркер (`ModelWorker`) последовательно обрабатывает задачи, имея эксклюзивный доступ к модели
  - Диспетчер результатов (`ResultDispatcher`) направляет ответы ожидающим запросам через `asyncio.Future`
  - Все операции с Future, которые могут вызвать колбэки, выносятся в отдельные потоки через `asyncio.to_thread`

- **Батчевая обработка** — поддержка пакетного кодирования нескольких текстов за один запрос для оптимальной производительности.

- **Гибкие таймауты** — клиент может указать желаемое время ожидания (`timeout` в теле запроса), сервис уважает этот таймаут (в пределах своего жесткого потолка).

- **JWT аутентификация** — все рабочие эндпоинты защищены JWT токенами с коротким сроком жизни (30 секунд).

- **Локальное хранение моделей** — модели скачиваются один раз и сохраняются локально в `models/sentence-transformers/`.

- **Защита от перегрузок** — очередь запросов с максимальным размером (1000 задач) и валидация размера входящих данных.

### API Эндпоинты

**Важно:** Все POST эндпоинты требуют JWT аутентификацию через заголовок `Authorization: Bearer <token>`.

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/info` | GET | Информация об энкодере и его модели |
| `/encode` | POST | Кодирование одного текста |
| `/encode_batch` | POST | Пакетное кодирование нескольких текстов |
| `/count_tokens` | POST | Подсчет токенов в одном тексте |
| `/count_tokens_batch` | POST | Подсчет токенов для нескольких текстов |
| `/vector_size` | GET | Размерность вектора модели |
| `/max_length` | GET | Максимальная длина текста в токенах |
| `/health` | GET | Проверка здоровья сервиса |

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
    "timeout": 60.0                     # опционально
}
```

Валидация:
- Количество текстов ≤ `MAX_BATCH_SIZE` (256)
- Каждый текст не пустой и ≤ `MAX_TEXT_LENGTH` (10000 символов)
- Суммарная длина ≤ `MAX_TOTAL_BATCH_LENGTH` (100000 символов)

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

Абсолютно верно! Это важный момент - документация должна объяснять, как работать с сервисом **без** использования готового клиента. Добавлю раздел:

### Аутентификация и подпись запросов

Все POST эндпоинты (`/encode`, `/encode_batch`, `/count_tokens`, `/count_tokens_batch`) требуют JWT-аутентификацию. Токен передается в заголовке `Authorization: Bearer <token>`.

#### 1. Формат токена

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
- `exp` должно быть не более чем через 30 секунд после `iat`
- Часы клиента и сервера должны быть синхронизированы (допустимое расхождение - несколько секунд)

#### 2. Генерация токена (примеры на разных языках)

##### Python
```python
import time
import jwt

SECRET_KEY = "your-32-char-secret-key-minimum"  # должен совпадать с INTERNAL_API_SECRET на сервере

def create_token(service_name: str = "my-service") -> str:
    current_time = int(time.time())
    payload = {
        "iss": service_name,
        "iat": current_time,
        "exp": current_time + 30,  # 30 секунд
        "service": service_name
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# Использование
token = create_token("rag-worker")
headers = {"Authorization": f"Bearer {token}"}
```

##### JavaScript/Node.js
```javascript
const jwt = require('jsonwebtoken');

const SECRET_KEY = 'your-32-char-secret-key-minimum';

function createToken(serviceName = 'my-service') {
    const currentTime = Math.floor(Date.now() / 1000);
    const payload = {
        iss: serviceName,
        iat: currentTime,
        exp: currentTime + 30,  // 30 секунд
        service: serviceName
    };
    return jwt.sign(payload, SECRET_KEY, { algorithm: 'HS256' });
}

// Использование
const token = createToken('rag-worker');
const headers = { 'Authorization': `Bearer ${token}` };
```

##### Go
```go
package main

import (
    "time"
    "github.com/golang-jwt/jwt/v5"
)

var secretKey = []byte("your-32-char-secret-key-minimum")

func createToken(serviceName string) (string, error) {
    currentTime := time.Now().Unix()
    claims := jwt.MapClaims{
        "iss":     serviceName,
        "iat":     currentTime,
        "exp":     currentTime + 30,
        "service": serviceName,
    }
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(secretKey)
}

// Использование
token, _ := createToken("rag-worker")
headers := map[string]string{"Authorization": "Bearer " + token}
```

##### Java
```java
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class JwtUtil {
    private static final String SECRET_KEY = "your-32-char-secret-key-minimum";
    
    public static String createToken(String serviceName) {
        long currentTime = System.currentTimeMillis() / 1000;
        
        Map<String, Object> claims = new HashMap<>();
        claims.put("iss", serviceName);
        claims.put("iat", currentTime);
        claims.put("exp", currentTime + 30);
        claims.put("service", serviceName);
        
        return Jwts.builder()
            .setClaims(claims)
            .signWith(SignatureAlgorithm.HS256, SECRET_KEY.getBytes())
            .compact();
    }
}

// Использование
String token = JwtUtil.createToken("rag-worker");
// Добавить в заголовок: "Authorization: Bearer " + token
```

##### curl (для тестирования)
```bash
# Сначала нужно сгенерировать токен (можно использовать Python одним выражением)
TOKEN=$(python3 -c "
import time, jwt
payload = {'iss': 'curl-client', 'iat': int(time.time()), 
           'exp': int(time.time()) + 30, 'service': 'curl-client'}
print(jwt.encode(payload, 'your-32-char-secret-key-minimum', algorithm='HS256'))
")

# Затем использовать в запросе
curl -X POST http://localhost:8260/encode \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text": "пример текста"}'
```

#### 3. Возможные ошибки и их обработка

| Код | Ответ | Причина | Действие |
|-----|-------|---------|----------|
| 403 | `{"detail": "Missing authentication token"}` | Отсутствует заголовок Authorization | Добавить токен |
| 403 | `{"detail": "Invalid token: Signature verification failed"}` | Неправильный секретный ключ или алгоритм | Проверить `INTERNAL_API_SECRET` |
| 403 | `{"detail": "Invalid token: Signature has expired"}` | Токен просрочен (exp < now) | Сгенерировать новый токен |
| 403 | `{"detail": "Invalid token: The specified alg value is not allowed"}` | Используется не HS256 | Проверить алгоритм подписи |

#### 4. Важные замечания

1. **Секретный ключ** должен быть одинаковым на всех сервисах, которые общаются друг с другом
2. **Синхронизация времени** - все сервисы должны иметь синхронизированные часы (NTP)
3. **Короткое время жизни** - 30 секунд достаточно для HTTP запроса, но слишком мало для ручного тестирования curl. Для отладки можно временно увеличить `JWT_EXPIRE_SECONDS` в коде сервиса
4. **Request ID** - рекомендуется добавлять уникальный ID запроса в токен для trace:
   ```python
   payload = {
       ...,
       "request_id": str(uuid.uuid4())[:8]
   }
   ```

#### 5. Без использования encoder-client

Если вы не используете готовый Python-клиент `encoder_client`, вам нужно:

1. **Сгенерировать JWT токен** для каждого запроса (токены живут 30 секунд)
2. **Добавить токен** в заголовок `Authorization: Bearer <token>`
3. **Отправить запрос** с правильным телом и заголовками
4. **Обработать ответ**, учитывая возможные 403 ошибки (истекший токен)

Пример полного цикла на Python без использования клиента:

```python
import httpx
import time
import jwt
from typing import List, Optional

class SimpleEncoderClient:
    def __init__(self, base_url: str, secret_key: str, service_name: str = "my-service"):
        self.base_url = base_url.rstrip('/')
        self.secret_key = secret_key
        self.service_name = service_name
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def _create_token(self) -> str:
        """Создает свежий JWT токен"""
        current_time = int(time.time())
        payload = {
            "iss": self.service_name,
            "iat": current_time,
            "exp": current_time + 30,
            "service": self.service_name
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    async def encode(self, text: str, request_type: str = "query") -> List[float]:
        """Кодирование одного текста"""
        url = f"{self.base_url}/encode"
        token = self._create_token()
        
        response = await self.client.post(
            url,
            json={"text": text, "request_type": request_type},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 403:
            # Токен мог истечь - можно повторить с новым токеном
            # Но в нашем случае токен свежий, значит проблема в ключе
            raise Exception(f"Auth failed: {response.json()}")
        
        response.raise_for_status()
        return response.json()["embedding"]
    
    async def close(self):
        await self.client.aclose()

# Использование
async def main():
    client = SimpleEncoderClient(
        base_url="http://localhost:8260",
        secret_key="your-32-char-secret-key-minimum"
    )
    
    embedding = await client.encode("пример текста")
    print(f"Получен вектор размерностью {len(embedding)}")
    
    await client.close()
```

Этот подход можно реализовать на любом языке программирования - главное правильно сформировать JWT токен.

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

Для разработки удобно использовать honcho, который запускает сервис с правильными параметрами:

```
# Procfile
encoder_service: uvicorn encoder_service.main:app \
  --port ${ENCODER_SERVICE_PORT} \
  --workers 1 \
  --lifespan on \
  --timeout-graceful-shutdown 10
```

Запуск с конкретным .env файлом:
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

Сборка образа:
```bash
# Сборка базового образа
docker build -f dockerfile.base -t encoder-base:latest .

# Сборка сервисного образа
docker build -f dockerfile.service -t encoder-service:latest .
```

#### Запуск нескольких экземпляров

```yaml
# docker-compose.yml
services:
  # Первый экземпляр (FRIDA)
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
      - .env          # общие параметры
      - .env.frida    # специфичные для FRIDA
    environment:
      - PORT=8260
    container_name: encoder_frida
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

  # Второй экземпляр (deepvk)
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
      --loop uvloop --lifespan on --timeout-graceful-shutdown 15
    init: true
    shm_size: '2gb'
    mem_limit: 8g
    cpus: '4.0'
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
| `INTERNAL_API_SECRET` | Секретный ключ для JWT | **Нет значения** |
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
| `MAX_TOTAL_BATCH_LENGTH` | Макс. суммарная длина | 100000 |
| **Пути** |
| `LOG_PATH` | Директория для логов | "logs" |
| `MODEL_PATH` | Директория для моделей | "models/sentence-transformers" |

#### Примеры .env файлов

**Общий .env:**
```bash
# Общие параметры для всех экземпляров
INTERNAL_API_SECRET=your-32-char-secret-key-minimum

# Таймауты
ENCODER_BASE_TIMEOUT=15
ENCODER_BATCH_TIMEOUT=60
MAX_SERVICE_TIMEOUT=120

# Лимиты
MAX_BATCH_SIZE=256
MAX_TEXT_LENGTH=10000
MAX_TOTAL_BATCH_LENGTH=100000
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
DEVICE=mps  # или cuda, или cpu
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

#### Логирование

Сервис использует структурированное логирование с разными уровнями:
- **DEBUG**: детальная информация о запросах, очередях
- **INFO**: запуск/остановка, успешные операции
- **WARNING**: проблемы с очередью, повторные попытки
- **ERROR**: ошибки модели, таймауты

Логи пишутся в `logs/encoder_service.log` и в stdout.

### Безопасность

- **JWT аутентификация** — все рабочие эндпоинты защищены
- **Короткое время жизни токена** — 30 секунд защищает от replay-атак
- **Очередь с ограничением** — защита от перегрузки (maxsize=1000)
- **Валидация входных данных** — Pydantic модели с проверкой границ
- **Изоляция контейнеров** — запуск под непривилегированным пользователем
- **Health checks** — для автоматического восстановления