# encoder_service/main.py

"""
Главный модуль Encoder Service.
Содержит FastAPI приложение, эндпоинты и логику жизненного цикла.
"""

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
import asyncio
from typing import Optional, Any
import uuid
import time

from shared.config import config
from shared.auth import require_auth
from shared.encoder_models import (
    # модели запросов
    EncodeRequest,
    BatchEncodeRequest,
    BatchTokenCountRequest,
    # модели ответов
    EncodeResponse,
    BatchEncodeResponse,
    TokenCountResponse,
    BatchTokenCountResponse,
    EncoderInfo,        # ответ эндпойнта /info
    EncoderModelInfo,   # используется внутри модели EncoderInfo
)

from encoder_service.worker import (
    ModelWorker, 
    Task, 
    TaskType, 
    BATCH_TASKS,
    clean_text
)
from encoder_service.dispatcher import ResultDispatcher


# ======================================================================
# КОНСТАНТЫ
# ======================================================================

# Жесткий потолок таймаута - сервис НИКОГДА не будет ждать дольше
# На случай, если клиент укажет в запросе какой-то нереально огромный тайм-аут
MAX_SERVICE_TIMEOUT = config.MAX_SERVICE_TIMEOUT


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
# Жизненный цикл приложения
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом FastAPI приложения.
    
    При запуске:
    1. Запускаем воркера (он загрузит модель)
    2. Запускаем диспетчер результатов
    3. Запускаем чистильщик зависших задач
    4. Ждем загрузки модели и формируем encoder_info
    
    При завершении:
    1. Корректно останавливаем все задачи
    """
    global encoder_info
    
    # Запускаем воркера
    worker_task = asyncio.create_task(worker.start())
    
    # Запускаем диспетчер результатов
    dispatcher_task = asyncio.create_task(dispatcher.dispatch(output_queue))
    
    # Запускаем чистильщик зависших задач
    cleaner_task = asyncio.create_task(
        dispatcher.cleanup_stale(MAX_SERVICE_TIMEOUT)
    )
    
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
    
    # Корректная остановка
    worker.running = False
    worker_task.cancel()
    dispatcher_task.cancel()
    cleaner_task.cancel()
    
    # Ждем завершения задач
    await asyncio.gather(worker_task, dispatcher_task, cleaner_task, return_exceptions=True)
    logger.info("Encoder Service остановлен")


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
    """
    Глобальный обработчик ошибок валидации запросов.
    
    Возвращает единый формат ошибки для всех случаев валидации.
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
# Функция отправки задачи (ядро логики таймаутов)
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
        
    Returns:
        Any: Результат обработки
        
    Raises:
        HTTPException: 503 при переполнении очереди, 504 при таймауте, 500 при ошибках
    """
    # Проверяем, что очередь не переполнена
    if input_queue.qsize() >= 900:
        logger.warning(f"Очередь почти полна: {input_queue.qsize()}")
        raise HTTPException(503, "Service busy, queue is full")
    
    # Генерируем уникальный ID задачи
    task_id = str(uuid.uuid4())
    
    # Создаем Future для ожидания результата
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
    
    # Регистрируем Future в диспетчере
    dispatcher.register(task_id, future, client_timeout)
    
    # Создаем задачу для воркера
    task = Task(
        task_id=task_id,
        task_type=task_type,
        data=data,
        created_at=time.time(),
        request_type=request_type,
        client_timeout=client_timeout
    )
    
    try:
        # Отправляем задачу в очередь воркера
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
# ЭНДПОЙНТЫ
# ======================================================================

@app.get("/info", response_model=EncoderInfo)
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_encoder_info(request: Request):
    """
    Возвращает полную информацию об этом экземпляре энкодера.
    
    Этот эндпоинт не требует аутентификации, так как используется
    клиентами при инициализации для получения данных о модели.
    """
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


@app.post("/encode", response_model=EncodeResponse)
@limiter.limit(config.RATE_LIMIT_ENCODE)
async def encode_text(
    encode_request: EncodeRequest, 
    request: Request,
    _: None = Depends(require_auth)
):
    """
    Кодирование одного текста в вектор.
    
    Новое опциональное поле: timeout - сколько клиент готов ждать (секунды)
    """
    try:
        cleaned_text = clean_text(encode_request.text)
        
        embedding = await submit_task(
            task_type=TaskType.ENCODE,
            data=cleaned_text,
            request_type=encode_request.request_type,
            client_timeout=encode_request.timeout
        )
        
        return EncodeResponse(
            embedding=embedding,
            service_available=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"encode endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.post("/encode_batch", response_model=BatchEncodeResponse)
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
        cleaned_texts = [clean_text(text) for text in batch_encode_request.texts]
        
        embeddings = await submit_task(
            task_type=TaskType.ENCODE_BATCH,
            data=cleaned_texts,
            request_type=batch_encode_request.request_type,
            client_timeout=batch_encode_request.timeout
        )
        
        return BatchEncodeResponse(
            embeddings=embeddings,
            service_available=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"encode_batch endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/vector_size")
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_vector_size(request: Request):
    """Возвращает размерность вектора используемой модели."""
    if worker.encoder:
        size = worker.encoder.get_sentence_embedding_dimension()
        return {"vector_size": size}
    return {"vector_size": None}


@app.get("/max_length")
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_max_length(request: Request):
    """Возвращает максимальную длину текста в токенах."""
    if worker.encoder:
        return {"max_length": worker.encoder.max_seq_length}
    return {"max_length": None}


@app.get("/health")
@limiter.limit(config.RATE_LIMIT_INFO)
async def health_check(request: Request):
    """
    Проверка здоровья сервиса.
    
    Возвращает статус сервиса, информацию о модели,
    размер очереди и количество активных запросов.
    """
    return {
        "status": "healthy" if worker.encoder else "degraded",
        "encoder_loaded": worker.encoder is not None,
        "queue_size": input_queue.qsize(),
        "active_requests": len(dispatcher.active_futures),
        "service_available": worker.encoder is not None,
        "max_timeout": MAX_SERVICE_TIMEOUT
    }


@app.post("/count_tokens", response_model=TokenCountResponse)
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS)
async def count_tokens(
    encode_request: EncodeRequest, 
    request: Request,
    _: None = Depends(require_auth)
):
    """Подсчет количества токенов в тексте."""
    try:
        tokens_count = await submit_task(
            task_type=TaskType.COUNT_TOKENS,
            data=encode_request.text,
            client_timeout=encode_request.timeout
        )
        
        # return {"tokens_count": tokens_count}
        return TokenCountResponse(
            tokens_count=tokens_count,
            service_available=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"count_tokens endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.post("/count_tokens_batch", response_model=BatchTokenCountResponse)
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS_BATCH)
async def count_tokens_batch(
    batch_token_count_request: BatchTokenCountRequest, 
    request: Request,
    _: None = Depends(require_auth)
):
    """Пакетный подсчет токенов для нескольких текстов."""
    try:
        tokens_counts = await submit_task(
            task_type=TaskType.COUNT_TOKENS_BATCH,
            data=batch_token_count_request.texts,
            client_timeout=batch_token_count_request.timeout
        )
        
        return BatchTokenCountResponse(
            tokens_counts=tokens_counts,
            service_available=True
        )
        
    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"count_tokens_batch endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )