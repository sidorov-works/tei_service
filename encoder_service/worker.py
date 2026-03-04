# encoder_service/worker.py

"""
Модуль воркера для Encoder Service.
Содержит класс ModelWorker - единственного владельца модели SentenceTransformer,
а также определения типов задач и структур данных для очереди.
"""

from shared.config import config
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path
import asyncio
from typing import Any, Optional, List
import time
from dataclasses import dataclass
from enum import Enum
import math

from shared.utils.logger import logger as base_logger, wrap_logger_methods

logger = wrap_logger_methods(base_logger, "ENCODER_SERVICE.WORKER")


# ======================================================================
# Вспомогательные функции для очистки данных
# ======================================================================

import re

_CLEAN_PATTERN = re.compile(r'[^\w\s.,!?:\-\'\"()]', flags=re.UNICODE)

def clean_text(text: str) -> str:
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

def clean_embedding(embedding: List[float]) -> List[float]:
    """
    Очищает вектор от NaN и Inf, заменяя их на 0.0.
    
    Args:
        embedding: Исходный вектор
        
    Returns:
        List[float]: Очищенный вектор
    """
    cleaned_embedding = []
    for x in embedding:
        if math.isnan(x) or math.isinf(x):
            cleaned_embedding.append(0.0)
        else:
            cleaned_embedding.append(x)
    return cleaned_embedding


# ======================================================================
# Типы задач и структуры данных
# ======================================================================

class TaskType(Enum):
    """
    Типы задач, которые может выполнять воркер.
    
    Каждый тип соответствует одному из эндпоинтов сервиса.
    """
    ENCODE = "encode"                     # /encode
    ENCODE_BATCH = "encode_batch"          # /encode_batch
    COUNT_TOKENS = "count_tokens"          # /count_tokens
    COUNT_TOKENS_BATCH = "count_tokens_batch"  # /count_tokens_batch


# Список батчевых операций - для них используется увеличенный таймаут
BATCH_TASKS = [TaskType.ENCODE_BATCH, TaskType.COUNT_TOKENS_BATCH]


@dataclass
class Task:
    """
    Задача для воркера.
    
    Каждая задача имеет уникальный ID, по которому диспетчер найдет Future,
    когда результат будет готов.
    
    Attributes:
        task_id: Уникальный идентификатор задачи (UUID)
        task_type: Тип задачи (encode, encode_batch, count_tokens, count_tokens_batch)
        data: Данные для обработки (текст или список текстов)
        created_at: Временная метка создания задачи (для расчета времени ожидания)
        request_type: Тип запроса для encode операций ("query" или "document")
        client_timeout: Сколько клиент готов ждать (опционально)
    """
    task_id: str
    task_type: TaskType
    data: Any  # Текст, список текстов и т.д.
    created_at: float
    request_type: Optional[str] = None
    client_timeout: Optional[float] = None  # Сколько клиент готов ждать


@dataclass
class TaskResult:
    """
    Результат выполнения задачи.
    
    Воркер кладет результат в очередь результатов, а диспетчер
    направляет его в соответствующий Future.
    
    Attributes:
        task_id: ID задачи, к которой относится результат
        success: Флаг успешности выполнения
        result: Результат (если success=True)
        error: Сообщение об ошибке (если success=False)
    """
    task_id: str
    success: bool
    result: Any = None
    error: str = None


class ModelWorker:
    """
    Единственный владелец модели SentenceTransformer.
    
    Этот класс работает в отдельной корутине и имеет эксклюзивный доступ к модели.
    Все запросы к модели проходят через него последовательно, что гарантирует
    потокобезопасность при работе с не-thread-safe моделью.
    
    Архитектура:
    - Воркер получает задачи из input_queue
    - Выполняет их одну за другой (модель не thread-safe!)
    - Результаты кладет в output_queue
    - Не знает ничего о FastAPI и клиентах - только задачи и результаты
    
    Attributes:
        input_queue: Очередь входящих задач от эндпоинтов
        output_queue: Очередь исходящих результатов для диспетчера
        encoder: Загруженная модель SentenceTransformer
        tokenizer: Токенизатор для подсчета токенов
        running: Флаг работы воркера
        tasks_processed: Счетчик обработанных задач (для статистики)
        total_processing_time: Суммарное время обработки (для статистики)
    """
    
    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        """
        Инициализация воркера.
        
        Args:
            input_queue: Очередь для получения задач от эндпоинтов
            output_queue: Очередь для отправки результатов диспетчеру
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.encoder = None
        self.tokenizer = None
        self.running = True
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        
    async def start(self):
        """
        Загружает модель и запускает основной цикл обработки задач.
        
        Этот метод должен быть запущен как asyncio.create_task().
        Он работает в бесконечном цикле, ожидая новые задачи из очереди.
        """
        try:
            model_id = config.HUGGING_FACE_MODEL_NAME
            model_path: Path = config.MODEL_PATH / model_id
            
            logger.info(f"Worker: загружаю модель {model_id}")
            
            # Загружаем SentenceTransformer модель (в отдельном потоке, чтобы не блокировать)
            self.encoder = await asyncio.to_thread(
                SentenceTransformer,
                str(model_path) if model_path.exists() else model_id,
                device=config.DEVICE
            )
            
            # Загружаем токенизатор
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                model_id
            )
            
            logger.info(f"Worker: модель загружена, начинаю обработку задач")
            
            # Основной цикл обработки задач
            while self.running:
                try:
                    # Ждем новую задачу (с таймаутом, чтобы можно было проверить running)
                    task = await asyncio.wait_for(
                        self.input_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Обрабатываем задачу
                    await self._process_task(task)
                    
                    # Отмечаем задачу как выполненную в очереди
                    self.input_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Нет задач - просто продолжаем цикл
                    continue
                except Exception as e:
                    logger.error(f"Worker: ошибка в основном цикле: {e}")
                    
        except Exception as e:
            logger.error(f"Worker: критическая ошибка при загрузке модели: {e}")
            self.running = False
    
    async def _process_task(self, task: Task):
        """
        Обрабатывает одну задачу.
        
        Это единственное место во всем сервисе, где используется модель.
        Метод последовательно вызывается для каждой задачи из очереди.
        
        Args:
            task: Задача для обработки
        """
        start_time = time.time()
        logger.debug(f"Worker: начал обработку задачи {task.task_id} типа {task.task_type}")
        
        try:
            result_data = None
            
            # В зависимости от типа задачи вызываем соответствующий метод модели
            if task.task_type == TaskType.ENCODE:
                # Одиночное кодирование
                text = task.data
                if task.request_type == "query":
                    text = config.QUERY_PREFIX + text
                elif task.request_type == "document":
                    text = config.DOCUMENT_PREFIX + text
                
                # Единственный вызов модели во всем сервисе!
                embedding = self.encoder.encode(text)
                result_data = clean_embedding(embedding.tolist())
                
            elif task.task_type == TaskType.ENCODE_BATCH:
                # Пакетное кодирование
                texts = task.data
                prefix = None
                if task.request_type == "query":
                    prefix = config.QUERY_PREFIX
                elif task.request_type == "document":
                    prefix = config.DOCUMENT_PREFIX
                
                if prefix:
                    texts = [prefix + text for text in texts]
                
                # Один вызов модели для всего батча!
                embeddings = self.encoder.encode(texts)
                result_data = [clean_embedding(emb.tolist()) for emb in embeddings]
                
            elif task.task_type == TaskType.COUNT_TOKENS:
                # Подсчет токенов в одном тексте
                text = task.data
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                result_data = len(tokens)
                
            elif task.task_type == TaskType.COUNT_TOKENS_BATCH:
                # Пакетный подсчет токенов
                texts = task.data
                token_counts = []
                for text in texts:
                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    token_counts.append(len(tokens))
                result_data = token_counts
            
            # Отправляем успешный результат в очередь результатов
            await self.output_queue.put(TaskResult(
                task_id=task.task_id,
                success=True,
                result=result_data
            ))
            
            # Обновляем статистику
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            logger.debug(f"Worker: завершил задачу {task.task_id} за {processing_time:.3f}с")
            
        except Exception as e:
            # В случае ошибки отправляем результат с ошибкой
            logger.error(f"Worker: ошибка при обработке задачи {task.task_id}: {e}")
            await self.output_queue.put(TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e)
            ))
    
    async def stop(self):
        """Останавливает воркер и освобождает ресурсы."""
        self.running = False
        logger.info(f"Worker: остановлен. Обработано задач: {self.tasks_processed}")