# encoder_service/main.py

"""
Главный модуль Encoder Service.
Предоставляет TEI-совместимый HTTP API для работы с эмбеддинговыми моделями.
"""

# Первым делом до любых импортов настроим логер
import setproctitle
from shared.utils.logger import logger as base_logger, wrap_logger_methods

setproctitle.setproctitle("encoder_service")
logger = wrap_logger_methods(base_logger, "ENCODER_SERVICE")

from shared.config import config
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import slowapi
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from typing import Optional, Any, Union, List
import uuid
import time

# Аутентификация через секретный заголовок 
from shared.auth_service import require_header_secret as require_auth

from shared.tei_models import (
    EmbedRequest,
    TokenizeRequest,
    EmbedResponse,
    TokenizeResponse,
    InfoResponse
)

from encoder_service.worker import (
    ModelWorker, 
    Task, 
    TaskType, 
    clean_text
)
from encoder_service.dispatcher import ResultDispatcher


# ======================================================================
# КОНСТАНТЫ
# ======================================================================

# Жесткий потолок таймаута - сервис НИКОГДА не будет ждать дольше
MAX_SERVICE_TIMEOUT = config.MAX_SERVICE_TIMEOUT


# ======================================================================
# Глобальные объекты
# ======================================================================

input_queue = asyncio.Queue(maxsize=1000)
output_queue = asyncio.Queue(maxsize=1000)

worker = ModelWorker(input_queue, output_queue)
dispatcher = ResultDispatcher()

# Информация о модели
model_info: Optional[InfoResponse] = None


# ======================================================================
# Жизненный цикл приложения
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом FastAPI приложения.
    """
    global model_info
    
    # Запускаем воркера
    worker_task = asyncio.create_task(worker.start())
    
    # Запускаем диспетчер результатов
    dispatcher_task = asyncio.create_task(dispatcher.dispatch(output_queue))
    
    # Запускаем чистильщик зависших задач
    cleaner_task = asyncio.create_task(
        dispatcher.cleanup_stale(MAX_SERVICE_TIMEOUT)
    )
    
    # Ждем загрузки модели
    while worker.encoder is None and worker.running:
        await asyncio.sleep(0.1)
    
    # Формируем информацию о модели в формате TEI
    if worker.encoder:
        model_info = InfoResponse(
            model_id=config.HUGGING_FACE_MODEL_NAME,
            max_input_length=worker.encoder.max_seq_length,
            dimension=worker.encoder.get_sentence_embedding_dimension(),
            status="operational"
        )
        logger.info(f"Encoder Service запущен, модель {config.HUGGING_FACE_MODEL_NAME} загружена")
    else:
        model_info = InfoResponse(
            model_id=config.HUGGING_FACE_MODEL_NAME,
            max_input_length=None,
            dimension=None,
            status="degraded"
        )
        logger.error("Модель не загрузилась!")
    
    yield
    
    # Корректная остановка
    worker.running = False
    worker_task.cancel()
    dispatcher_task.cancel()
    cleaner_task.cancel()
    
    await dispatcher.close()
    await asyncio.gather(worker_task, dispatcher_task, cleaner_task, return_exceptions=True)
    logger.info("Encoder Service остановлен")


# ======================================================================
# FastAPI приложение
# ======================================================================

app = FastAPI(
    lifespan=lifespan, 
    title="Encoder Service",
    description="TEI-совместимый сервис для кодирования текста в векторные представления"
)

# Rate limiter
limiter = slowapi.Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, slowapi._rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Глобальный обработчик ошибок валидации запросов.
    """
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
# Функция отправки задачи
# ======================================================================

async def submit_task(
    task_type: TaskType, 
    data: Any, 
    request_type: Optional[str] = None
) -> Any:
    """
    Отправка задачи воркеру и ожидание результата.
    
    TEI не поддерживает передачу таймаутов в теле запроса,
    поэтому используем только таймауты сервиса по умолчанию.
    
    Args:
        task_type: Тип задачи
        data: Данные для обработки
        request_type: query или document (для encode операций)
        
    Returns:
        Any: Результат обработки
        
    Raises:
        HTTPException: 503 при переполнении очереди, 504 при таймауте
    """
    # Проверяем, что очередь не переполнена
    if input_queue.qsize() >= 900:
        logger.warning(f"Очередь почти полна: {input_queue.qsize()}")
        raise HTTPException(503, "Service busy, queue is full")
    
    task_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    
    # Определяем таймаут в зависимости от типа задачи
    default_timeout = (
        config.ENCODER_BATCH_TIMEOUT 
        if task_type in [TaskType.ENCODE_BATCH, TaskType.COUNT_TOKENS_BATCH]
        else config.ENCODER_BASE_TIMEOUT
    )
    effective_timeout = min(default_timeout, MAX_SERVICE_TIMEOUT)
    
    # Регистрируем Future без клиентского таймаута
    dispatcher.register(task_id, future, None)
    
    task = Task(
        task_id=task_id,
        task_type=task_type,
        data=data,
        created_at=time.time(),
        request_type=request_type,
        client_timeout=None  # TEI не поддерживает клиентские таймауты
    )
    
    try:
        await input_queue.put(task)
        
        try:
            result = await asyncio.wait_for(future, timeout=effective_timeout)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут задачи {task_id} после {effective_timeout}с")
            dispatcher.unregister(task_id)
            if not future.done():
                future.cancel()
            raise HTTPException(504, f"Request timeout after {effective_timeout} seconds")
            
    except Exception as e:
        dispatcher.unregister(task_id)
        if not future.done():
            future.cancel()
        logger.error(f"Ошибка при обработке задачи {task_id}: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")


# ======================================================================
# TEI-СОВМЕСТИМЫЕ ЭНДПОЙНТЫ
# ======================================================================

@app.get("/info", response_model=InfoResponse)
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_encoder_info(request: Request):
    """
    TEI-совместимый endpoint информации о модели.
    
    Returns:
        {
            "model_id": "deepvk/USER2-base",
            "max_input_length": 8192,
            "dimension": 768,
            "status": "operational"
        }
    """
    if not worker.encoder:
        return InfoResponse(
            model_id=config.HUGGING_FACE_MODEL_NAME,
            max_input_length=None,
            dimension=None,
            status="degraded"
        )
    
    # Обновляем актуальные значения из модели
    return InfoResponse(
        model_id=config.HUGGING_FACE_MODEL_NAME,
        max_input_length=worker.encoder.max_seq_length,
        dimension=worker.encoder.get_sentence_embedding_dimension(),
        status="operational"
    )


@app.post("/embed")
@limiter.limit(config.RATE_LIMIT_ENCODE)
async def embed(
    request: Request,
    embed_request: EmbedRequest,
    _: None = Depends(require_auth)
):
    """
    TEI-совместимый endpoint для получения эмбеддингов.
    
    Поддерживает как одиночные тексты, так и батчи.
    
    Пример запроса (single):
    {
        "inputs": "What is Deep Learning?",
        "prompt_name": "query",
        "normalize": true,
        "truncate": true
    }
    
    Пример запроса (batch):
    {
        "inputs": ["text1", "text2"],
        "prompt_name": "document",
        "normalize": true
    }
    
    Returns (single):
    {
        "embedding": [0.123, -0.456, ...]
    }
    
    Returns (batch):
    {
        "embeddings": [[0.123, ...], [0.789, ...]]
    }
    """
    try:
        # Определяем тип запроса из prompt_name
        # TEI использует prompt_name для выбора шаблона
        request_type = embed_request.prompt_name or "query"
        
        # Обработка одиночного текста
        if isinstance(embed_request.inputs, str):
            cleaned_text = clean_text(embed_request.inputs)
            
            embedding = await submit_task(
                task_type=TaskType.ENCODE,
                data=cleaned_text,
                request_type=request_type
            )
            
            return {"embedding": embedding}
        
        # Обработка батча
        else:
            cleaned_texts = [clean_text(text) for text in embed_request.inputs]
            
            embeddings = await submit_task(
                task_type=TaskType.ENCODE_BATCH,
                data=cleaned_texts,
                request_type=request_type
            )
            
            return {"embeddings": embeddings}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"embed endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.post("/tokenize")
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS)
async def tokenize(
    request: Request,
    tokenize_request: TokenizeRequest,
    _: None = Depends(require_auth)
):
    """
    TEI-совместимый endpoint для подсчета токенов.
    
    Пример запроса (single):
    {
        "inputs": "What is Deep Learning?",
        "add_special_tokens": true,
        "truncate": true
    }
    
    Пример запроса (batch):
    {
        "inputs": ["text1", "text2"],
        "add_special_tokens": true
    }
    
    Returns (single):
    {
        "tokens_count": 8
    }
    
    Returns (batch):
    {
        "tokens_counts": [5, 8]
    }
    """
    try:
        # Обработка одиночного текста
        if isinstance(tokenize_request.inputs, str):
            tokens_count = await submit_task(
                task_type=TaskType.COUNT_TOKENS,
                data=tokenize_request.inputs,
                request_type=None
            )
            
            return {"tokens_count": tokens_count}
        
        # Обработка батча
        else:
            tokens_counts = await submit_task(
                task_type=TaskType.COUNT_TOKENS_BATCH,
                data=tokenize_request.inputs,
                request_type=None
            )
            
            return {"tokens_counts": tokens_counts}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"tokenize endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/health")
@limiter.limit(config.RATE_LIMIT_INFO)
async def health_check(request: Request):
    """
    Проверка здоровья сервиса.
    """
    active_count = await dispatcher.get_active_count()
    
    return {
        "status": "healthy" if worker.encoder else "degraded",
        "encoder_loaded": worker.encoder is not None,
        "queue_size": input_queue.qsize(),
        "active_requests": active_count,
        "service_available": worker.encoder is not None,
        "max_timeout": MAX_SERVICE_TIMEOUT
    }