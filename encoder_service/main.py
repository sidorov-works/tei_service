# encoder_service/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from shared.utils.logger import logger as base_logger, wrap_logger_methods
from shared.config import config
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


# Модели данных для валидации запросов
class EncodeRequest(BaseModel):
    """
    Для передачи энкодеру одного текста текста
    """
    text: str
    request_type: Optional[str] = "query"

class BatchEncodeRequest(BaseModel):
    """Запрос на пакетное кодирование"""
    texts: List[str]
    request_type: Optional[str] = "query"


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

    async def connect(self) -> bool:
        """Инициализация сервиса - загрузка модели эмбеддингов"""
        try:
            model_data = config.EMBEDDING_MODEL
            model_id = model_data['model']
            
            model_path = Path(config.MODELS_PATH) / model_data['subdir'] / model_id
            
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
            logger.info(f"Использую устройство: {model_data['device']}")
            
            # 2. Загружаем SentenceTransformer модель с диска
            self.encoder = await asyncio.to_thread(
                SentenceTransformer,
                str(model_path),
                device=model_data['device']
            )

            # 3. Проверяем наличие токенизатора локально SentenceTransformer 
            # сохраняет модель в своей структуре, а токенизатор нужно сохранять отдельно
            tokenizer_path = model_path / "tokenizer"
            
            if tokenizer_path.exists() and (tokenizer_path / "tokenizer_config.json").exists():
                # Загружаем токенизатор из локальной копии
                logger.info(f"Загружаю токенизатор из локальной копии: {tokenizer_path}")
                self.tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    str(tokenizer_path)
                )
            else:
                # Скачиваем токенизатор из интернета и сохраняем локально
                logger.info(f"Загружаю токенизатор из интернета: {model_id}")
                self.tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    model_id
                )
                
                # Сохраняем токенизатор локально для будущих запусков
                tokenizer_path.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(self.tokenizer.save_pretrained, str(tokenizer_path))
                logger.info(f"Токенизатор сохранен в {tokenizer_path}")
            
            # 4. Проверяем загрузку
            if self.tokenizer is None:
                logger.error("Не удалось загрузить токенизатор")
                return False
                
            self.service_available = True
            logger.info(f"Модель энкодера успешно загружена")
            logger.info(f"Токенизатор загружен, размер словаря: {self.tokenizer.vocab_size}")
            
            # 5. Проверяем соответствие max_seq_length в конфиге
            if hasattr(self.encoder, 'max_seq_length'):
                config_max_len = model_data.get('max_seq_length', 512)
                if self.encoder.max_seq_length != config_max_len:
                    logger.warning(
                        f"Модель имеет max_seq_length={self.encoder.max_seq_length}, "
                        f"но в конфиге указано {config_max_len}. "
                        f"Используется значение модели."
                    )
            
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

def _clean_text(text: str) -> str:
    """
    Очистка текста перед кодированием.
    
    Удаляет нестандартные символы, которые могут вызвать проблемы
    с токенизатором или кодированием.
    """
    if not text:
        return ""
    
    # Оставляем только буквы, цифры, пробелы и основные знаки препинания
    cleaned_text = re.sub(r'[^\w\s.,!?:\-\'\"()]', '', text, flags=re.UNICODE)
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

@app.post("/encode")
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
                cleaned_text = config.EMBEDDING_MODEL.get("query_prefix", "search_query: ") + cleaned_text
            elif request.request_type == "document":
                cleaned_text = config.EMBEDDING_MODEL.get("document_prefix", "search_document: ") + cleaned_text
        
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
                prefix = config.EMBEDDING_MODEL.get("query_prefix", "search_query: ")
            elif request.request_type == "document":
                prefix = config.EMBEDDING_MODEL.get("document_prefix", "search_document: ")
        
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
def get_vector_size():
    """
    Возвращает длину вектора в используемой модели
    """
    return {"vector_size": config.EMBEDDING_MODEL.get("vector_size", 0)}

@app.get("/max_length")
def get_max_length():
    """
    Возвращает максимальную длину текста (в токенах)
    """
    return {"max_length": config.EMBEDDING_MODEL.get("max_seq_length", 0)}

@app.get("/health")
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

@app.get("/status")
async def status():
    """
    Детальный статус сервиса.
    
    Можно оставить async, не использует модель.
    """
    return {
        "service": "Encoder Service",
        "status": "operational" if encoder_service.service_available else "degraded",
        "encoder_loaded": encoder_service.encoder is not None,
        "model": config.EMBEDDING_MODEL.get("model", "unknown"),
        "device": config.EMBEDDING_MODEL.get("device", "cpu"),
        "note": "All encode endpoints are synchronous for thread safety"
    }

@app.post("/count_tokens")
def get_tokens_count(request: EncodeRequest, _: None = Depends(verify_internal)):
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