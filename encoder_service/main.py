"""
Главный модуль Encoder Service.
Предоставляет TEI-совместимый HTTP API для работы с эмбеддинговыми моделями.
"""

# Настройка логгера должна быть первым делом
from logger_utils import configure_root, get_logger
from pathlib import Path
from shared.config import config
configure_root(
    level=config.LOGGING_LEVEL, 
    log_file=config.LOG_PATH / "app.log",
    fmt=config.LOG_FORMAT
)

logger = get_logger("ENCODER_SERVICE")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import slowapi
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from typing import Optional, Any, List
import uuid
import time

import setproctitle
setproctitle.setproctitle("encoder_service")

from shared.auth_service import require_header_secret as require_auth
from shared.tei_models import (
    EmbedRequest,
    TokenizeRequest,
    TokenInfo,
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
# Глобальные объекты
# ======================================================================

input_queue = asyncio.Queue(maxsize=config.INPUT_QUEUE_MAXSIZE)
output_queue = asyncio.Queue(maxsize=config.OUTPUT_QUEUE_MAXSIZE)

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
    cleaner_task = asyncio.create_task(dispatcher.cleanup_stale())
    
    # Ждем загрузки модели
    while worker.encoder is None and worker.running:
        await asyncio.sleep(0.1)
    
    # Формируем информацию о модели в формате TEI
    if worker.encoder:
        model_info = InfoResponse(
            model_id=config.HUGGING_FACE_MODEL_NAME,
            max_input_length=worker.encoder.max_seq_length,
            max_client_batch_size=config.MAX_SERVICE_BATCH_SIZE
        )
        logger.info(f"Encoder Service запущен, модель {config.HUGGING_FACE_MODEL_NAME} загружена")
    else:
        model_info = InfoResponse(
            model_id=config.HUGGING_FACE_MODEL_NAME,
            max_input_length=None,
            max_client_batch_size=config.MAX_SERVICE_BATCH_SIZE
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
    description="TEI-подобный сервис для кодирования текста в векторные представления"
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
    request_type: Optional[str] = None,
    task_timeout: float = None,  # обязательный параметр, должен передаваться из эндпоинта
    normalize: bool = True,
    truncate: bool = True,
) -> Any:
    """
    Отправка задачи воркеру и ожидание результата.
    
    Args:
        task_type: Тип задачи
        data: Данные для обработки
        request_type: query или document (для encode операций)
        task_timeout: Максимальное время выполнения операции (из конфига)
        
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
    
    # Регистрируем Future с таймаутом операции
    dispatcher.register(task_id, future, task_timeout)
    
    task = Task(
        task_id=task_id,
        task_type=task_type,
        data=data,
        created_at=time.time(),
        request_type=request_type,
        truncate=truncate,
        normalize=normalize
    )
    
    try:
        await input_queue.put(task)
        
        try:
            result = await asyncio.wait_for(future, timeout=task_timeout)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут задачи {task_id} после {task_timeout} сек.")
            dispatcher.unregister(task_id)
            if not future.done():
                future.cancel()
            raise HTTPException(504, f"Request timeout after {task_timeout} seconds")
            
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
    TEI-совместимый endpoint информации о модели
    """
    if not worker.encoder:
        return InfoResponse(
            model_id=config.HUGGING_FACE_MODEL_NAME,
            max_input_length=None,
            max_client_batch_size=config.MAX_SERVICE_BATCH_SIZE
        )
    
    return InfoResponse(
        model_id=config.HUGGING_FACE_MODEL_NAME,
        max_input_length=worker.encoder.max_seq_length,
        max_client_batch_size=config.MAX_SERVICE_BATCH_SIZE
    )


@app.post("/embed", response_model=None)
@limiter.limit(config.RATE_LIMIT_EMBED)
async def embed(
    request: Request,
    embed_request: EmbedRequest,
    _: None = Depends(require_auth)
):
    """
    TEI-совместимый endpoint для получения эмбеддингов.
    
    Возвращает список списков float, даже для одного текста.
    """
    try:
        request_type = embed_request.prompt_name or "query"
        
        if isinstance(embed_request.inputs, str):
            cleaned_text = clean_text(embed_request.inputs)
            
            embeddings: List[List[float]] = await submit_task(
                task_type=TaskType.ENCODE,
                data=cleaned_text,
                request_type=request_type,
                task_timeout=config.EMBED_TIMEOUT,
                normalize=embed_request.normalize,
                truncate=embed_request.truncate,
            )
            
            return embeddings
        
        else:
            cleaned_texts = [clean_text(text) for text in embed_request.inputs]
            
            embeddings: List[List[float]] = await submit_task(
                task_type=TaskType.ENCODE_BATCH,
                data=cleaned_texts,
                request_type=request_type,
                task_timeout=config.EMBED_TIMEOUT
            )
            
            return embeddings
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"embed endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.post("/tokenize", response_model=None)
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS)
async def tokenize(
    request: Request,
    tokenize_request: TokenizeRequest,
    _: None = Depends(require_auth)
):
    """
    TEI-совместимый endpoint для токенизации.
    
    Возвращает список списков TokenInfo, даже для одного текста.
    """
    try:
        if isinstance(tokenize_request.inputs, str):
            inputs = [tokenize_request.inputs]
        else:
            inputs = tokenize_request.inputs
            
        tokens_info_batch: List[List[TokenInfo]] = await submit_task(
            task_type=TaskType.TOKENIZE,
            data=inputs,
            task_timeout=config.TOKENIZE_TIMEOUT
        )
        
        return tokens_info_batch
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"tokenize endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/health")
@limiter.limit(config.RATE_LIMIT_HEALTH)
async def health_check(request: Request):
    """
    Проверка здоровья сервиса.
    
    Returns:
        - 200 OK с пустым телом, если сервис здоров
        - 503 Service Unavailable, если сервис не может обрабатывать запросы
    """
    if worker.encoder is None:
        logger.warning("Health check failed: encoder not loaded")
        return Response(status_code=503)
    
    if input_queue.qsize() > config.INPUT_QUEUE_MAXSIZE * config.HEALTH_QUEUE_THRESHOLD:
        logger.warning(f"Health check failed: input queue size {input_queue.qsize()}")
        return Response(status_code=503)
    
    if output_queue.qsize() > config.OUTPUT_QUEUE_MAXSIZE * config.HEALTH_QUEUE_THRESHOLD:
        logger.warning(f"Health check failed: output queue size {output_queue.qsize()}")
        return Response(status_code=503)
    
    return Response(status_code=200)