# encoder_service/main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import slowapi

from shared.utils.logger import logger as base_logger, wrap_logger_methods
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
from pydantic import BaseModel
from pathlib import Path
import setproctitle
import asyncio
from typing import List, Optional
import re
import math

import torch  # для очистки памяти в методе close()

setproctitle.setproctitle("encoder_service")
logger = wrap_logger_methods(base_logger, "ENCODER_SERVICE")


async def verify_internal(request: Request):
    """
    Проверка (по секретному заголовку), 
    что вызов внутренних эндпойнтов является действительно внутренним. 
    В аргументы функций эндпойнтов необходимо включить: 
    ```_: None = fastapi.Depends(verify_internal)
    """
    internal_secret = request.headers.get("X-Internal-Secret")
    expected_secret = config.INTERNAL_API_SECRET
    
    if not internal_secret or internal_secret != expected_secret:
        logger.warning(f"Invalid internal secret from {request.client.host}")
        raise HTTPException(status_code=403, detail="Forbidden")

class EncoderService:
    """
    Сервис кодирования текста в эмбеддинги.
    
    КРИТИЧЕСКИ ВАЖНО: SentenceTransformer модели НЕ thread-safe.
    Нельзя вызывать encoder.encode() параллельно из разных потоков.
    
    Решение: использовать синхронные эндпоинты в FastAPI.
    FastAPI сам будет ставить запросы в очередь и обрабатывать их по одному.
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
                device=config.DEVICE  # напрямую из конфига
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
            # Явное удаление модели
            del self.encoder
            self.encoder = None
            
            # Очистка памяти GPU/MPS
            if torch.backends.mps.is_available():  # для Mac M1/M2
                torch.mps.empty_cache()
                logger.info("MPS cache cleared")
            elif torch.cuda.is_available():        # для NVIDIA GPU
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

# Создаем limiter с in-memory хранилищем (по умолчанию)
limiter = slowapi.Limiter(key_func=slowapi.util.get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    slowapi.errors.RateLimitExceeded, 
    slowapi._rate_limit_exceeded_handler
)

# Сделаем более красивую обработку ошибок валидации запросов (400 и с понятным сообщением)
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
    
    БЕЗ этого обработчика:
    - FastAPI вернул бы стандартный ответ с массивом ошибок
    - Клиент получил бы технический ответ, неудобный для парсинга
    
    С ЭТИМ обработчиком:
    1. FastAPI при возникновении RequestValidationError 
       вызывает эту функцию (благодаря декоратору)
    2. Мы преобразуем массив ошибок в понятную строку
    3. Добавляем service_available для единообразия API
    4. Возвращаем единый формат ошибок для всех эндпоинтов
    
    Returns:
        JSONResponse с HTTP 400 и единым форматом ошибки
    """
    errors = []
    for error in exc.errors():
        # 'loc' содержит путь к полю с ошибкой (например: ["body", "texts", 0])
        field = " -> ".join(str(x) for x in error["loc"])
        msg = error["msg"]
        errors.append(f"{field}: {msg}")
    
    error_msg = "; ".join(errors)
    
    # Логируем для отладки (видно в консоли)
    logger.warning(f"Validation error: {error_msg}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error",
            "detail": error_msg,
            "service_available": encoder_service.service_available
        }
    )

# Компилируем один раз при загрузке модуля.
_CLEAN_PATTERN = re.compile(r'[^\w\s.,!?:\-\'\"()]', flags=re.UNICODE)

def _clean_text(text: str) -> str:
    """
    Очистка текста перед кодированием.
    
    Удаляет нестандартные символы, которые могут вызвать проблемы
    с токенизатором или кодированием.
    """
    if not text:
        return ""
    # Оставляем только буквы, цифры, пробелы и основные знаки препинания
    cleaned_text = _CLEAN_PATTERN.sub('', text)
    # Убираем лишние пробелы и табуляции
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
async def get_encoder_info(request: Request):
    """
    Возвращает полную информацию об этом экземпляре энкодера.
    
    Этот эндпоинт не требует аутентификации, так как используется
    клиентами при инициализации для получения данных о модели.
    Информация о модели не является чувствительной.
    
    Returns:
        EncoderInfo: Информация об энкодере и его модели
    """
    if not encoder_service.service_available or not encoder_service.encoder_info:
        # Используем fallback с именем из конфига
        return EncoderInfo(
            name=config.ENCODER_NAME,  # берем из config, а не из сервиса
            model_info=EncoderModelInfo(
                name=config.HUGGING_FACE_MODEL_NAME,
                vector_size=None,
                max_seq_length=None,
                query_prefix=config.QUERY_PREFIX,
                document_prefix=config.DOCUMENT_PREFIX
            ),
            status="degraded"
        )
    
    # Возвращаем encoder_info
    return encoder_service.encoder_info

@app.post("/encode")
@limiter.limit(config.RATE_LIMIT_ENCODE)
def encode_text(request: EncodeRequest, _: None = Depends(verify_internal)):
    """
    Кодирование одного текста в вектор.
    
    ВНИМАНИЕ: Этот эндпоинт СИНХРОННЫЙ.
    
    Почему синхронный:
    1. SentenceTransformer.encode() не thread-safe
    2. FastAPI для синхронных эндпоинтов ставит запросы в очередь
    3. Нет риска параллельных вызовов модели
    
    Если w_classifier и w_rag вызовут одновременно:
    - Первый запрос начнет обрабатываться
    - Второй будет ждать в очереди FastAPI
    - Когда первый завершится, начнется второй
    """
    try:
        if not encoder_service.service_available:
            return {"error": "Encoder service not available", "service_available": False}
        
        # Очищаем текст
        cleaned_text = _clean_text(request.text)

        # Добавляем префикс в зависимости от укзанного типа запроса
        if request.request_type:
            if request.request_type == "query":
                cleaned_text = config.QUERY_PREFIX + cleaned_text
            elif request.request_type == "document":
                cleaned_text = config.DOCUMENT_PREFIX + cleaned_text
        
        # СИНХРОННЫЙ вызов модели
        # FastAPI гарантирует, что в этот момент модель не используется другими запросами
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
def encode_batch(request: BatchEncodeRequest, _: None = Depends(verify_internal)):
    """
    Пакетное кодирование нескольких текстов.
    
    ВНИМАНИЕ: Этот эндпоинт СИНХРОННЫЙ.
    
    Это основной эндпоинт для воркеров (w_classifier и w_rag).
    
    Как работает очередность:
    1. w_classifier отправляет запрос → FastAPI начинает обработку
    2. w_rag отправляет запрос → FastAPI ставит в очередь
    3. w_classifier запрос обработан → ответ
    4. w_rag запрос начинает обработку → ответ
    
    Результат: никаких segmentation fault, запросы обрабатываются по очереди.
    """
    try:
        if not encoder_service.service_available:
            return {"error": "Encoder service not available", "service_available": False}
        
        # Префикс в зависимости от укзанного типа запроса
        prefix = None
        if request.request_type:
            if request.request_type == "query":
                prefix = config.QUERY_PREFIX
            elif request.request_type == "document":
                prefix = config.DOCUMENT_PREFIX
        
        # Очищаем все тексты и добавляем префикс (если нужен)
        if prefix:
            cleaned_texts = [prefix + _clean_text(text) for text in request.texts]
        else:
            cleaned_texts = [_clean_text(text) for text in request.texts]
        
        # Логируем информацию о батче для отладки
        total_chars = sum(len(text) for text in cleaned_texts)
        logger.debug(f"Processing batch: {len(cleaned_texts)} texts, {total_chars} total chars")
        
        # СИНХРОННЫЙ вызов модели для всего батча
        # Модель умеет обрабатывать батчи эффективно, но только по одному батчу за раз
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
def get_vector_size():
    """
    Возвращает длину вектора в используемой модели
    """
    return {"vector_size": encoder_service.encoder_info.model_info.vector_size}

@app.get("/max_length")
@limiter.limit(config.RATE_LIMIT_INFO)
def get_max_length():
    """
    Возвращает максимальную длину текста (в токенах)
    """
    return {"max_length": encoder_service.encoder_info.model_info.max_seq_length}

@app.get("/health")
@limiter.limit(config.RATE_LIMIT_INFO)
async def health_check():
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
def tokens_count(request: EncodeRequest, _: None = Depends(verify_internal)):
    """
    Определение количества токенов в тексте.
    
    ВНИМАНИЕ: Этот эндпоинт СИНХРОННЫЙ.
    
    Токенизатор также не является полностью thread-safe,
    поэтому используем синхронный эндпоинт.
    """
    try:
        if not encoder_service.service_available:
            return {"error": "Encoder service not available", "service_available": False}
        
        tokens = encoder_service.tokenizer.encode(
            request.text,
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
def count_tokens_batch_sync(
    request: BatchTokenCountRequest, 
    _: None = Depends(verify_internal)
):
    """
    СИНХРОННЫЙ подсчет токенов для нескольких текстов за один запрос.
    
    Возвращает СПИСОК длин для каждого текста в том же порядке.
    
    Пример запроса:
    {
        "texts": ["короткий текст", "очень длинный текст с множеством слов и предложений"]
    }
    
    Пример ответа:
    {
        "tokens_counts": [3, 15],  # ← первый текст - 3 токена, второй - 15 токенов
        "count": 2,
        "service_available": true
    }
    """
    try:
        if not encoder_service.service_available:
            return {
                "tokens_counts": [],  # пустой список при ошибке
                "count": 0,
                "service_available": False
            }
        
        token_counts = []
        
        # Для КАЖДОГО текста считаем токены и добавляем в список
        for text in request.texts:
            tokens = encoder_service.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True
            )
            token_counts.append(len(tokens))  # ← добавляем длину в список
        
        logger.debug(f"Batch token count: processed {len(request.texts)} texts")
        
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