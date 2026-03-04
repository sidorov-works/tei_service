# encoder_service/main.py

# Сразу настроим логер
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
from typing import List, Optional
import re
import math
import threading
from functools import wraps
import torch

# Проверка подписи клиента в эндпойнтах.
# Использование в эндпоинтах:
#     @app.post("/encode")
#     async def encode(..., token_data: dict = Depends(verify_jwt_token)):
#         service_name = token_data.get("service")
# Пока оставляем GET-эндпойнты без проверки (считаем что это безопасно)
from shared.auth import verify_jwt_token, require_auth


# Глобальная блокировка для защиты модели от параллельных вызовов

# SentenceTransformer модели НЕ потокобезопасны. Нельзя вызывать encoder.encode()
# параллельно из разных потоков - это приведет к segmentation fault.
# FastAPI запускает пул из 40 потоков для синхронных эндпоинтов,
# поэтому необходима принудительная синхронизация.
_MODEL_LOCK = threading.Lock()

def synchronized_endpoint(func):
    """
    Декоратор для синхронизации эндпоинтов, работающих с моделью.
    
    Как работает:
    1. FastAPI вызывает эндпоинт в одном из потоков пула (до 40 потоков)
    2. Декоратор перехватывает вызов и захватывает глобальную блокировку _model_lock
    3. Если блокировка свободна - поток продолжает выполнение эндпоинта
    4. Если блокировка занята - поток ждет на входе в with
    5. Когда текущий поток освобождает блокировку, следующий поток в очереди просыпается
    
    Таким образом, даже при 40 параллельных запросах, модель всегда
    используется только одним потоком в каждый момент времени.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _MODEL_LOCK:
            return func(*args, **kwargs)
    return wrapper


class EncoderService:
    """
    Сервис кодирования текста в эмбеддинги.
    
    КРИТИЧЕСКИ ВАЖНО: SentenceTransformer модели НЕ thread-safe.
    Защита от параллельных вызовов реализована через декоратор synchronized_endpoint
    на уровне эндпоинтов. Сам класс не содержит методов-оберток и не управляет блокировками.
    """
    def __init__(self):
        self.encoder = None   # Модель SentenceTransformer
        self.tokenizer = None # Отдельный объект токенизатора
        self.service_available = False  # Флаг доступности сервиса
        self.encoder_info: Optional[EncoderInfo] = None # Заполним, когда загрузим модель

    async def connect(self) -> bool:
        """Инициализация сервиса - загрузка модели эмбеддингов"""
        try:
            model_id = config.HUGGING_FACE_MODEL_NAME
            model_path: Path = config.MODEL_PATH / model_id
            
            # 1. Проверяем и скачиваем SentenceTransformer модель, если её нет
            if not model_path.exists():
                logger.info(f"Модель не найдена по пути {model_path}. Начинаю скачивание...")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Скачиваем модель из интернета
                tmp_model = await asyncio.to_thread(
                    SentenceTransformer, 
                    model_id
                )
                # Сохраняем локально
                await asyncio.to_thread(tmp_model.save, str(model_path))
                logger.info(f"Модель сохранена в {model_path}")
            else:
                logger.info(f"Найдена локальная копия модели: {model_path}")

            logger.info(f"Загружаю модель с диска: {model_path}")
            logger.info(f"Использую устройство: {config.DEVICE}")
            
            # 2. Загружаем SentenceTransformer модель с диска
            self.encoder = await asyncio.to_thread(
                SentenceTransformer,
                str(model_path),
                device=config.DEVICE
            )

            # 3. Загружаем токенизатор
            tokenizer_path = model_path / "tokenizer"
            
            if tokenizer_path.exists() and (tokenizer_path / "tokenizer_config.json").exists():
                logger.info(f"Загружаю токенизатор из локальной копии: {tokenizer_path}")
                self.tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    str(tokenizer_path)
                )
            else:
                logger.info(f"Загружаю токенизатор из интернета: {model_id}")
                self.tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    model_id
                )
                
                tokenizer_path.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(self.tokenizer.save_pretrained, str(tokenizer_path))
                logger.info(f"Токенизатор сохранен в {tokenizer_path}")
            
            # 4. Проверяем загрузку
            if self.tokenizer is None:
                logger.error("Не удалось загрузить токенизатор")
                return False
            
            # 5. Получаем реальные значения параметров из модели
            actual_vector_size = self.encoder.get_sentence_embedding_dimension()
            actual_max_length = self.encoder.max_seq_length
            
            self.encoder_info = EncoderInfo(
                name=config.ENCODER_NAME,
                model_info=EncoderModelInfo(
                    name=config.HUGGING_FACE_MODEL_NAME,
                    vector_size=actual_vector_size,
                    max_seq_length=actual_max_length,
                    query_prefix=config.QUERY_PREFIX,
                    document_prefix=config.DOCUMENT_PREFIX
                )
            )

            self.service_available = True
            logger.info(f"Модель энкодера успешно загружена: {self.encoder_info.model_info.name}")
            logger.info(f"Размер вектора: {self.encoder_info.model_info.vector_size}")
            logger.info(f"Max sequence length: {self.encoder_info.model_info.max_seq_length}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}", exc_info=True)
            self.service_available = False
            return False

    async def close(self):
        """Корректное завершение работы с очисткой ресурсов"""
        if self.encoder:
            del self.encoder
            self.encoder = None
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("MPS cache cleared")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        
        self.service_available = False
        logger.info("Encoder service shutdown complete")


# Глобальный экземпляр сервиса
encoder_service = EncoderService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом FastAPI приложения.
    
    При запуске: загружаем модель.
    При завершении: очищаем ресурсы.
    """
    try:
        if not await encoder_service.connect():
            logger.error("Encoder service failed to initialize")
        yield
    finally:
        await encoder_service.close()


app = FastAPI(
    lifespan=lifespan, 
    title="Encoder Service",
    description="Сервис для кодирования текста в векторные представления"
)

# Создаем rate limiter с in-memory хранилищем
limiter = slowapi.Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, 
    slowapi._rate_limit_exceeded_handler
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Глобальный обработчик ошибок валидации запросов.
    
    В FastAPI ВСЕ входящие запросы проходят через валидацию:
    1. Парсинг JSON тела запроса
    2. Проверка соответствия типам данных (Pydantic)
    3. Выполнение пользовательских валидаторов (@field_validator)
    
    Если на любом из этих этапов возникает ошибка,
    FastAPI выбрасывает исключение RequestValidationError.
    
    Returns:
        JSONResponse с HTTP 400 и единым форматом ошибки
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
            "service_available": encoder_service.service_available
        }
    )


# Компилируем один раз при загрузке модуля
_CLEAN_PATTERN = re.compile(r'[^\w\s.,!?:\-\'\"()]', flags=re.UNICODE)

def _clean_text(text: str) -> str:
    """
    Очистка текста перед кодированием.
    
    Удаляет нестандартные символы, которые могут вызвать проблемы
    с токенизатором или кодированием.
    """
    if not text:
        return ""
    cleaned_text = _CLEAN_PATTERN.sub('', text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


def _clean_embedding(embedding: List[float]) -> Optional[List[float]]:
    """
    Очищает вектор от NaN и Inf
    """
    cleaned_embedding = []
    for x in embedding:
        if math.isnan(x):
            cleaned_embedding.append(0)
        elif math.isinf(x):
            raise Exception("Inf value in embedding")
        else:
            cleaned_embedding.append(x)
    return cleaned_embedding


@app.get("/info", response_model=EncoderInfo)
@limiter.limit(config.RATE_LIMIT_INFO)
async def get_encoder_info(
        request: Request # нужно для rate limiting (формальность)
    ):
    """
    Возвращает полную информацию об этом экземпляре энкодера.
    
    Этот эндпоинт не требует аутентификации, так как используется
    клиентами при инициализации для получения данных о модели.
    Информация о модели не является чувствительной.
    
    Returns:
        EncoderInfo: Информация об энкодере и его модели
    """
    if not encoder_service.service_available or not encoder_service.encoder_info:
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
    
    return encoder_service.encoder_info


@app.post("/encode")
@limiter.limit(config.RATE_LIMIT_ENCODE)
@synchronized_endpoint
def encode_text(
    encode_request: EncodeRequest, 
    request: Request, # нужно для rate limiting
    _: None = Depends(require_auth)
    ):
    """
    Кодирование одного текста в вектор.
    
    ВНИМАНИЕ: Этот эндпоинт СИНХРОННЫЙ.
    
    Почему синхронный:
    1. SentenceTransformer.encode() не thread-safe
    2. FastAPI для синхронных эндпоинтов использует пул потоков (до 40 потоков)
    3. Без защиты 40 потоков могут одновременно вызвать модель -> SEGFAULT
    4. Декоратор @synchronized_endpoint захватывает глобальную блокировку
    5. Только один поток может выполнять encode() в каждый момент времени
    
    Как работает последовательность:
    - Первый запрос захватывает блокировку и начинает encode()
    - Второй запрос пытается захватить блокировку и ждет
    - Третий запрос тоже ждет
    - Когда первый завершается, второй захватывает блокировку и выполняет encode()
    - И так далее
    
    Таким образом, запросы обрабатываются строго последовательно,
    что гарантирует безопасную работу с не-потокобезопасной моделью.
    """
    try:
        if not encoder_service.service_available:
            return {"error": "Encoder service not available", "service_available": False}
        
        cleaned_text = _clean_text(encode_request.text)

        if encode_request.request_type:
            if encode_request.request_type == "query":
                cleaned_text = config.QUERY_PREFIX + cleaned_text
            elif encode_request.request_type == "document":
                cleaned_text = config.DOCUMENT_PREFIX + cleaned_text
        
        # Прямой вызов модели - декоратор гарантирует, что этот код
        # выполняется только одним потоком в каждый момент времени
        embedding = encoder_service.encoder.encode(cleaned_text)
        cleaned_embedding = _clean_embedding(embedding.tolist()) 
        
        return {
            "embedding": cleaned_embedding,
            "dimension": len(embedding),
            "service_available": True
        }
        
    except Exception as e:
        logger.error(f"Encode endpoint error: {e}")
        return {
            "error": str(e),
            "service_available": encoder_service.service_available
        }


@app.post("/encode_batch")
@limiter.limit(config.RATE_LIMIT_ENCODE_BATCH)
@synchronized_endpoint
def encode_batch(
    batch_encode_request: BatchEncodeRequest, 
    request: Request, # нужно для rate limiting
    _: None = Depends(require_auth)
    ):
    """
    Пакетное кодирование нескольких текстов.
    
    ВАЖНО: Использует ТУ ЖЕ САМУЮ блокировку, что и /encode.
    Это значит, что:
    - Не могут одновременно выполняться /encode и /encode_batch
    - Все операции с моделью строго последовательны
    - Batch-обработка не дает выигрыша в параллельности, но дает
      выигрыш в эффективности использования GPU (один большой батч
      вместо множества маленьких)
    
    Как работает при высокой нагрузке:
    1. Приходит 10 запросов к /encode и 5 запросов к /encode_batch
    2. Все они пытаются захватить одну блокировку _model_lock
    3. Выполняются строго по очереди: encode, encode_batch, encode, ...
    4. Модель всегда используется одним потоком, race conditions исключены
    """
    try:
        if not encoder_service.service_available:
            return {"error": "Encoder service not available", "service_available": False}
        
        prefix = None
        if batch_encode_request.request_type:
            if batch_encode_request.request_type == "query":
                prefix = config.QUERY_PREFIX
            elif batch_encode_request.request_type == "document":
                prefix = config.DOCUMENT_PREFIX
        
        if prefix:
            cleaned_texts = [prefix + _clean_text(text) for text in batch_encode_request.texts]
        else:
            cleaned_texts = [_clean_text(text) for text in batch_encode_request.texts]
        
        total_chars = sum(len(text) for text in cleaned_texts)
        logger.debug(f"Processing batch: {len(cleaned_texts)} texts, {total_chars} total chars")
        
        # Прямой batch-вызов модели под защитой той же блокировки
        embeddings = encoder_service.encoder.encode(cleaned_texts)
        embeddings_list = [_clean_embedding(embedding.tolist()) for embedding in embeddings]
        
        return {
            "embeddings": embeddings_list,
            "count": len(embeddings_list),
            "service_available": True
        }
        
    except Exception as e:
        logger.error(f"Encode batch endpoint error: {e}")
        return {
            "error": str(e),
            "service_available": encoder_service.service_available
        }


@app.get("/vector_size")
@limiter.limit(config.RATE_LIMIT_INFO)
def get_vector_size(
        request: Request, # нужно для rate limiting
    ):
    """
    Возвращает длину вектора в используемой модели
    """
    return {"vector_size": encoder_service.encoder_info.model_info.vector_size}


@app.get("/max_length")
@limiter.limit(config.RATE_LIMIT_INFO)
def get_max_length(
        request: Request, # нужно для rate limiting
    ):
    """
    Возвращает максимальную длину текста (в токенах)
    """
    return {"max_length": encoder_service.encoder_info.model_info.max_seq_length}


@app.get("/health")
@limiter.limit(config.RATE_LIMIT_INFO)
async def health_check(
        request: Request, # нужно для rate limiting
    ):
    """
    Проверка здоровья сервиса.
    
    Этот эндпоинт можно оставить async, так как он
    не использует модель и вызывается редко.
    """
    return {
        "status": "healthy" if encoder_service.service_available else "degraded",
        "encoder_loaded": encoder_service.encoder is not None,
        "service_available": encoder_service.service_available
    }


@app.post("/count_tokens")
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS)
@synchronized_endpoint
def count_tokens(
        encode_request: EncodeRequest, 
        request: Request, # нужно для rate limiting
        _: None = Depends(require_auth)
    ):
    """
    Определение количества токенов в тексте.
    
    Токенизатор также не является полностью потокобезопасным,
    поэтому защищаем его той же блокировкой, что и модель.
    """
    try:
        if not encoder_service.service_available:
            return {"error": "Encoder service not available", "service_available": False}
        
        tokens = encoder_service.tokenizer.encode(
            encode_request.text,
            add_special_tokens=True,
            truncation=True
        )
        return {"tokens_count": len(tokens)}

    except Exception as e:
        logger.error(f"Count_tokens endpoint error: {e}")
        return {
            "error": str(e),
            "service_available": encoder_service.service_available
        }


@app.post("/count_tokens_batch", response_model=BatchTokenCountResponse)
@limiter.limit(config.RATE_LIMIT_COUNT_TOKENS_BATCH)
@synchronized_endpoint
def count_tokens_batch(
    batch_token_count_request: BatchTokenCountRequest, 
    request: Request, # нужно для rate limiting
    _: None = Depends(require_auth)
):
    """
    Пакетный подсчет токенов для нескольких текстов за один запрос.

    
    Возвращает СПИСОК длин для каждого текста в том же порядке.
    
    Пример запроса:
    {
        "texts": ["короткий текст", "очень длинный текст"]
    }
    
    Пример ответа:
    {
        "tokens_counts": [3, 15],
        "count": 2,
        "service_available": true
    }
    """
    try:
        if not encoder_service.service_available:
            return {
                "tokens_counts": [],
                "count": 0,
                "service_available": False
            }
        
        token_counts = []
        
        for text in batch_token_count_request.texts:
            tokens = encoder_service.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True
            )
            token_counts.append(len(tokens))
        
        logger.debug(f"Batch token count: processed {len(batch_token_count_request.texts)} texts")
        
        return {
            "tokens_counts": token_counts, 
            "count": len(token_counts),
            "service_available": True
        }
        
    except Exception as e:
        logger.error(f"Count_tokens_batch endpoint error: {e}")
        return {
            "tokens_counts": [],
            "count": 0,
            "service_available": encoder_service.service_available
        }