# encoder_service/main.py

import setproctitle
from shared.utils.logger import logger as base_logger, wrap_logger_methods

setproctitle.setproctitle("encoder_service")
logger = wrap_logger_methods(base_logger, "ENCODER_SERVICE")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import slowapi
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from shared.config import config
from shared.encoder_models import (
    EncoderInfo, 
    EncoderModelInfo,
    EncodeRequest,
    BatchEncodeRequest,
    BatchTokenCountRequest,
    BatchTokenCountResponse
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path
import asyncio
from typing import List, Optional, Dict, Any
import re
import math
import uuid
import time
from dataclasses import dataclass
from enum import Enum

from shared.auth import verify_jwt_token, require_auth


# ======================================================================
# КОНСТАНТЫ
# ======================================================================

# Жесткий потолок таймаута - сервис НИКОГДА не будет ждать дольше
# На случай, если клиент укажет в запросе какой-то нереально 
# огромный тайм-аут
MAX_SERVICE_TIMEOUT = config.MAX_SERVICE_TIMEOUT


# ======================================================================
# АРХИТЕКТУРА: Очередь + Воркер
# ======================================================================

class TaskType(Enum):
    """Типы задач, которые может выполнять воркер"""
    ENCODE = "encode"
    ENCODE_BATCH = "encode_batch"
    COUNT_TOKENS = "count_tokens"
    COUNT_TOKENS_BATCH = "count_tokens_batch"

BATCH_TASKS = [TaskType.ENCODE_BATCH, TaskType.COUNT_TOKENS_BATCH]

@dataclass
class Task:
    """
    Задача для воркера.
    
    Каждая задача имеет уникальный ID, по которому мы найдем Future,
    когда результат будет готов.
    
    Важное дополнение: client_timeout - сколько клиент готов ждать.
    Воркер может использовать это для приоритезации, но основная логика
    таймаутов реализована в submit_task.
    """
    task_id: str
    task_type: TaskType
    data: Any  # Текст, список текстов и т.д.
    created_at: float
    request_type: Optional[str] = None
    client_timeout: Optional[float] = None  # Сколько клиент готов ждать

@dataclass
class TaskResult:
    """Результат выполнения задачи"""
    task_id: str
    success: bool
    result: Any = None
    error: str = None

class ModelWorker:
    """
    Единственный владелец модели SentenceTransformer.
    
    Работает в отдельной корутине, имеет эксклюзивный доступ к модели.
    Все запросы к модели проходят через него последовательно.
    """
    
    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.encoder = None
        self.tokenizer = None
        self.running = True
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        
    async def start(self):
        """Загружает модель и запускает основной цикл обработки"""
        try:
            model_id = config.HUGGING_FACE_MODEL_NAME
            model_path: Path = config.MODEL_PATH / model_id
            
            logger.info(f"Worker: загружаю модель {model_id}")
            self.encoder = await asyncio.to_thread(
                SentenceTransformer,
                str(model_path) if model_path.exists() else model_id,
                device=config.DEVICE
            )
            
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                model_id
            )
            
            logger.info(f"Worker: модель загружена, начинаю обработку задач")
            
            while self.running:
                try:
                    # Ждем новую задачу
                    task = await asyncio.wait_for(
                        self.input_queue.get(), 
                        timeout=1.0
                    )
                    
                    await self._process_task(task)
                    self.input_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Worker: ошибка в основном цикле: {e}")
                    
        except Exception as e:
            logger.error(f"Worker: критическая ошибка при загрузке модели: {e}")
            self.running = False
    
    async def _process_task(self, task: Task):
        """Обрабатывает одну задачу - единственное место, где используется модель"""
        
        start_time = time.time()
        logger.debug(f"Worker: начал обработку задачи {task.task_id} типа {task.task_type}")
        
        try:
            result_data = None
            
            if task.task_type == TaskType.ENCODE:
                text = task.data
                if task.request_type == "query":
                    text = config.QUERY_PREFIX + text
                elif task.request_type == "document":
                    text = config.DOCUMENT_PREFIX + text
                
                embedding = self.encoder.encode(text)
                result_data = _clean_embedding(embedding.tolist())
                
            elif task.task_type == TaskType.ENCODE_BATCH:
                texts = task.data
                prefix = None
                if task.request_type == "query":
                    prefix = config.QUERY_PREFIX
                elif task.request_type == "document":
                    prefix = config.DOCUMENT_PREFIX
                
                if prefix:
                    texts = [prefix + text for text in texts]
                
                embeddings = self.encoder.encode(texts)
                result_data = [_clean_embedding(emb.tolist()) for emb in embeddings]
                
            elif task.task_type == TaskType.COUNT_TOKENS:
                text = task.data
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                result_data = len(tokens)
                
            elif task.task_type == TaskType.COUNT_TOKENS_BATCH:
                texts = task.data
                token_counts = []
                for text in texts:
                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    token_counts.append(len(tokens))
                result_data = token_counts
            
            # Отправляем успешный результат
            await self.output_queue.put(TaskResult(
                task_id=task.task_id,
                success=True,
                result=result_data
            ))
            
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            logger.debug(f"Worker: завершил задачу {task.task_id} за {processing_time:.3f}с")
            
        except Exception as e:
            logger.error(f"Worker: ошибка при обработке задачи {task.task_id}: {e}")
            await self.output_queue.put(TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e)
            ))
    
    async def stop(self):
        """Останавливает воркер"""
        self.running = False
        logger.info(f"Worker: остановлен. Обработано задач: {self.tasks_processed}")


class ResultDispatcher:
    """
    Диспетчер результатов.
    
    Получает результаты от воркера и направляет их в соответствующие Future.
    """
    
    def __init__(self):
        # Словарь активных Future: task_id -> (future, created_at, client_timeout)
        # client_timeout нужен для проверки, не пора ли убить задачу
        self.active_futures: Dict[str, tuple[asyncio.Future, float, Optional[float]]] = {}
        self._lock = asyncio.Lock()
        
    def register(self, task_id: str, future: asyncio.Future, client_timeout: Optional[float] = None):
        """Регистрирует новый ожидающий запрос"""
        self.active_futures[task_id] = (future, time.time(), client_timeout)
        
    def unregister(self, task_id: str):
        """Удаляет завершенный или отмененный запрос"""
        self.active_futures.pop(task_id, None)
        
    async def dispatch(self, result_queue: asyncio.Queue):
        """
        Основной цикл диспетчера.
        Слушает очередь результатов и пробуждает ожидающие запросы.
        """
        while True:
            try:
                result = await result_queue.get()
                
                future_info = self.active_futures.get(result.task_id)
                if future_info:
                    future, _, _ = future_info
                    
                    if result.success:
                        future.set_result(result.result)
                    else:
                        future.set_exception(Exception(result.error))
                    
                    self.unregister(result.task_id)
                else:
                    logger.warning(f"Получен результат для неизвестного task_id: {result.task_id}")
                
                result_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatcher: ошибка: {e}")
    
    async def cleanup_stale(self):
        """
        Периодическая очистка "зависших" задач.
        
        Проверяет, не превысили ли задачи максимальное время ожидания.
        Учитывает client_timeout - если клиент просил ждать 60 секунд,
        а прошло 70, то задачу пора убивать.
        """
        while True:
            await asyncio.sleep(10)  # Проверяем каждые 10 секунд
            now = time.time()
            stale_ids = []
            
            async with self._lock:
                for task_id, (future, created_at, client_timeout) in list(self.active_futures.items()):
                    # Сколько прошло времени с создания задачи
                    age = now - created_at
                    
                    # Определяем максимальное время жизни задачи
                    if client_timeout:
                        # Если клиент указал таймаут - уважаем его (но не больше MAX_SERVICE_TIMEOUT)
                        max_age = min(client_timeout, MAX_SERVICE_TIMEOUT)
                    else:
                        # Если не указал - используем базовый таймаут encode
                        # Но мы не знаем тип задачи в диспетчере, поэтому используем консервативное значение
                        max_age = config.ENCODER_BASE_TIMEOUT
                    
                    # Добавляем запас в 5 секунд на обработку
                    max_age += 5
                    
                    if age > max_age and not future.done():
                        stale_ids.append(task_id)
                        # Отменяем Future с таймаутом
                        future.set_exception(
                            asyncio.TimeoutError(f"Task expired after {age:.1f} seconds")
                        )
                
                for task_id in stale_ids:
                    self.active_futures.pop(task_id, None)
            
            if stale_ids:
                logger.warning(f"Очищено зависших задач: {len(stale_ids)}")


# ======================================================================
# Глобальные объекты
# ======================================================================

# Очереди для общения между FastAPI и воркером
input_queue = asyncio.Queue(maxsize=1000)
output_queue = asyncio.Queue(maxsize=1000)

# Воркер с моделью
worker = ModelWorker(input_queue, output_queue)

# Диспетчер результатов
dispatcher = ResultDispatcher()

# Информация об энкодере
encoder_info: Optional[EncoderInfo] = None


# ======================================================================
# Вспомогательные функции
# ======================================================================

_CLEAN_PATTERN = re.compile(r'[^\w\s.,!?:\-\'\"()]', flags=re.UNICODE)

def _clean_text(text: str) -> str:
    """Очистка текста перед кодированием"""
    if not text:
        return ""
    cleaned_text = _CLEAN_PATTERN.sub('', text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def _clean_embedding(embedding: List[float]) -> List[float]:
    """Очищает вектор от NaN и Inf"""
    cleaned_embedding = []
    for x in embedding:
        if math.isnan(x):
            cleaned_embedding.append(0.0)
        elif math.isinf(x):
            cleaned_embedding.append(0.0)
        else:
            cleaned_embedding.append(x)
    return cleaned_embedding


# ======================================================================
# Жизненный цикл приложения
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder_info
    
    # Запускаем воркера
    worker_task = asyncio.create_task(worker.start())
    
    # Запускаем диспетчер и чистильщик
    dispatcher_task = asyncio.create_task(dispatcher.dispatch(output_queue))
    cleaner_task = asyncio.create_task(dispatcher.cleanup_stale())
    
    # Ждем загрузки модели без таймаута - сколько надо, столько и ждем
    while worker.encoder is None and worker.running:
        await asyncio.sleep(0.1)
    
    # Теперь модель точно загружена
    if worker.encoder:
        encoder_info = EncoderInfo(
            name=config.ENCODER_NAME,
            model_info=EncoderModelInfo(
                name=config.HUGGING_FACE_MODEL_NAME,
                vector_size=worker.encoder.get_sentence_embedding_dimension(),
                max_seq_length=worker.encoder.max_seq_length,
                query_prefix=config.QUERY_PREFIX,
                document_prefix=config.DOCUMENT_PREFIX
            ),
            status="operational"
        )
        logger.info(f"Encoder Service запущен, модель {config.HUGGING_FACE_MODEL_NAME} загружена")
    else:
        encoder_info = EncoderInfo(
            name=config.ENCODER_NAME,
            model_info=EncoderModelInfo(
                name=config.HUGGING_FACE_MODEL_NAME,
                vector_size=None,
                max_seq_length=None,
                query_prefix=config.QUERY_PREFIX,
                document_prefix=config.DOCUMENT_PREFIX
            ),
            status="degraded"
        )
        logger.error("Модель не загрузилась!")
    
    logger.info(f"Максимальный таймаут сервиса: {MAX_SERVICE_TIMEOUT}с")
    
    yield
    
    # Останавливаем
    worker.running = False
    worker_task.cancel()
    dispatcher_task.cancel()
    cleaner_task.cancel()


# ======================================================================
# FastAPI приложение
# ======================================================================

app = FastAPI(
    lifespan=lifespan, 
    title="Encoder Service",
    description="Сервис для кодирования текста в векторные представления"
)

# Rate limiter
limiter = slowapi.Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, slowapi._rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Обработчик ошибок валидации"""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        msg = error["msg"]
        errors.append(f"{field}: {msg}")
    
    error_msg = "; ".join(errors)
    logger.warning(f"Validation error: {error_msg}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error",
            "detail": error_msg,
            "service_available": True
        }
    )


# ======================================================================
# Функция отправки задачи (САМАЯ ВАЖНАЯ - ЗДЕСЬ ЛОГИКА ТАЙМАУТОВ)
# ======================================================================

async def submit_task(
    task_type: TaskType, 
    data: Any, 
    request_type: Optional[str] = None,
    client_timeout: Optional[float] = None
) -> Any:
    """
    Отправка задачи воркеру и ожидание результата с учетом таймаутов.
    
    Логика таймаутов:
    1. Если клиент передал timeout - используем его, но не больше MAX_SERVICE_TIMEOUT
    2. Если клиент не передал - используем таймаут по умолчанию из конфига
    3. В любом случае, никогда не ждем дольше MAX_SERVICE_TIMEOUT
    
    Args:
        task_type: Тип задачи
        data: Данные для обработки
        request_type: query или document
        client_timeout: Сколько клиент готов ждать (опционально)
    """
    # Проверяем очередь
    if input_queue.qsize() >= 900:
        logger.warning(f"Очередь почти полна: {input_queue.qsize()}")
        raise HTTPException(503, "Service busy, queue is full")
    
    # Генерируем ID задачи
    task_id = str(uuid.uuid4())
    
    # Создаем Future
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    
    # Определяем эффективный таймаут
    # 1. Базовый таймаут из конфига (зависит от типа задачи)
    default_timeout = (
        config.ENCODER_BATCH_TIMEOUT 
        if task_type in BATCH_TASKS 
        else config.ENCODER_BASE_TIMEOUT
    )
    
    # 2. Если клиент передал свой таймаут - берем его, но не больше MAX_SERVICE_TIMEOUT
    if client_timeout is not None:
        effective_timeout = min(client_timeout, MAX_SERVICE_TIMEOUT)
        logger.debug(f"Клиент указал timeout={client_timeout}, используем {effective_timeout}")
    else:
        # 3. Иначе используем базовый, но тоже не больше MAX_SERVICE_TIMEOUT
        effective_timeout = min(default_timeout, MAX_SERVICE_TIMEOUT)
        logger.debug(f"Клиент не указал timeout, используем {effective_timeout}")
    
    # Регистрируем в диспетчере (передаем client_timeout для cleaner'a)
    dispatcher.register(task_id, future, client_timeout)
    
    # Создаем задачу
    task = Task(
        task_id=task_id,
        task_type=task_type,
        data=data,
        created_at=time.time(),
        request_type=request_type,
        client_timeout=client_timeout  # Передаем воркеру для информации
    )
    
    try:
        # Отправляем задачу
        await input_queue.put(task)
        logger.debug(f"Задача {task_id} отправлена в очередь")
        
        # Ждем результат с эффективным таймаутом
        try:
            result = await asyncio.wait_for(future, timeout=effective_timeout)
            return result
            
        except asyncio.TimeoutError:
            # Таймаут - клиент устал ждать
            logger.warning(f"Таймаут задачи {task_id} после {effective_timeout}с")
            dispatcher.unregister(task_id)
            
            # Отменяем Future, чтобы освободить ресурсы
            if not future.done():
                future.cancel()
            
            # Возвращаем 504 с информацией о таймауте
            raise HTTPException(
                504, 
                f"Request timeout after {effective_timeout} seconds"
            )
            
        except asyncio.CancelledError:
            # Клиент отменил запрос (разорвал соединение)
            logger.debug(f"Запрос {task_id} отменен клиентом")
            dispatcher.unregister(task_id)
            if not future.done():
                future.cancel()
            raise
            
    except HTTPException:
        # Пробрасываем HTTP исключения дальше
        raise
    except Exception as e:
        # Любые другие ошибки
        dispatcher.unregister(task_id)
        if not future.done():
            future.cancel()
        logger.error(f"Ошибка при обработке задачи {task_id}: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")


# ======================================================================
# ЭНДПОЙНТЫ (интерфейс полностью сохранен + новое поле timeout)
# ======================================================================

@app.get("/info", response_model=EncoderInfo)
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_encoder_info(request: Request):
    """Информация об энкодере"""
    if not worker.encoder:
        return EncoderInfo(
            name=config.ENCODER_NAME,
            model_info=EncoderModelInfo(
                name=config.HUGGING_FACE_MODEL_NAME,
                vector_size=None,
                max_seq_length=None,
                query_prefix=config.QUERY_PREFIX,
                document_prefix=config.DOCUMENT_PREFIX
            ),
            status="degraded"
        )
    
    # Обновляем информацию из модели
    encoder_info.model_info.vector_size = worker.encoder.get_sentence_embedding_dimension()
    encoder_info.model_info.max_seq_length = worker.encoder.max_seq_length
    
    return encoder_info


@app.post("/encode")
@limiter.limit(config.RATE_LIMIT_ENCODE)
async def encode_text(
    encode_request: EncodeRequest, 
    request: Request,
    _: None = Depends(require_auth)
):
    """
    Кодирование одного текста.
    
    Новое опциональное поле: timeout - сколько клиент готов ждать (секунды)
    """
    try:
        cleaned_text = _clean_text(encode_request.text)
        
        embedding = await submit_task(
            task_type=TaskType.ENCODE,
            data=cleaned_text,
            request_type=encode_request.request_type,
            client_timeout=encode_request.timeout  # Передаем таймаут клиента
        )
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "service_available": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Encode endpoint error: {e}")
        return {
            "error": str(e),
            "service_available": worker.encoder is not None
        }


@app.post("/encode_batch")
@limiter.limit(config.RATE_LIMIT_ENCODE_BATCH)
async def encode_batch(
    batch_encode_request: BatchEncodeRequest, 
    request: Request,
    _: None = Depends(require_auth)
):
    """
    Пакетное кодирование нескольких текстов.
    
    Новое опциональное поле: timeout - сколько клиент готов ждать (секунды)
    """
    try:
        cleaned_texts = [_clean_text(text) for text in batch_encode_request.texts]
        
        embeddings = await submit_task(
            task_type=TaskType.ENCODE_BATCH,
            data=cleaned_texts,
            request_type=batch_encode_request.request_type,
            client_timeout=batch_encode_request.timeout  # Передаем таймаут клиента
        )
        
        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "service_available": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Encode batch endpoint error: {e}")
        return {
            "error": str(e),
            "service_available": worker.encoder is not None
        }


@app.get("/vector_size")
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_vector_size(request: Request):
    """Размерность вектора"""
    if worker.encoder:
        size = worker.encoder.get_sentence_embedding_dimension()
        return {"vector_size": size}
    return {"vector_size": None}


@app.get("/max_length")
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_max_length(request: Request):
    """Максимальная длина текста"""
    if worker.encoder:
        return {"max_length": worker.encoder.max_seq_length}
    return {"max_length": None}


@app.get("/health")
@limiter.limit(config.RATE_LIMIT_INFO)
async def health_check(request: Request):
    """Проверка здоровья"""
    return {
        "status": "healthy" if worker.encoder else "degraded",
        "encoder_loaded": worker.encoder is not None,
        "queue_size": input_queue.qsize(),
        "active_requests": len(dispatcher.active_futures),
        "service_available": worker.encoder is not None,
        "max_timeout": MAX_SERVICE_TIMEOUT
    }


@app.post("/count_tokens")
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS)
async def count_tokens(
    encode_request: EncodeRequest, 
    request: Request,
    _: None = Depends(require_auth)
):
    """Подсчет токенов в тексте"""
    try:
        token_count = await submit_task(
            task_type=TaskType.COUNT_TOKENS,
            data=encode_request.text,
            client_timeout=encode_request.timeout  # И здесь тоже
        )
        
        return {"tokens_count": token_count}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Count_tokens endpoint error: {e}")
        return {
            "error": str(e),
            "service_available": worker.tokenizer is not None
        }


@app.post("/count_tokens_batch", response_model=BatchTokenCountResponse)
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS_BATCH)
async def count_tokens_batch(
    batch_token_count_request: BatchTokenCountRequest, 
    request: Request,
    _: None = Depends(require_auth)
):
    """Пакетный подсчет токенов"""
    try:
        token_counts = await submit_task(
            task_type=TaskType.COUNT_TOKENS_BATCH,
            data=batch_token_count_request.texts,
            client_timeout=batch_token_count_request.timeout  # И здесь
        )
        
        return {
            "tokens_counts": token_counts,
            "count": len(token_counts),
            "service_available": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Count_tokens_batch endpoint error: {e}")
        return {
            "tokens_counts": [],
            "count": 0,
            "service_available": worker.tokenizer is not None
        }