"""
Модуль воркера для Encoder Service.
Содержит класс ModelWorker - единственного владельца модели SentenceTransformer,
а также определения типов задач и структур данных для очереди.
"""

import os
# Запрещаем внутреннюю многопоточность токенизаторов, чтобы избежать проблем с fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from shared.config import config
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import asyncio
from typing import Any, Optional, List
import time
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import re

import logging
logger = logging.getLogger(__name__)


# ======================================================================
# Вспомогательные функции для очистки данных
# ======================================================================

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
    """
    ENCODE = "encode"                   # /embed для одного текста
    ENCODE_BATCH = "encode_batch"       # /embed для батча
    TOKENIZE = "tokenize"               # /tokenize и для одного текста, и для батча


@dataclass
class Task:
    """
    Задача для воркера.
    
    Attributes:
        task_id: Уникальный идентификатор задачи (UUID)
        task_type: Тип задачи
        data: Данные для обработки (текст или список текстов)
        created_at: Временная метка создания задачи
        prompt_name: Имя промпта для добавления префикса запроса
        truncate: Обрезать ли тексты длиннее max_input_length
        normalize: Нормализовать ли векторы 
    """
    task_id: str
    task_type: TaskType
    data: Any
    created_at: float
    # request_type: Optional[str] = None
    prompt_name: Optional[str] = None
    truncate: bool = True # только для /embed
    normalize: bool = False # только для /embed


@dataclass
class TaskResult:
    """
    Результат выполнения задачи.
    
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
    
    Все запросы к модели проходят через него последовательно, что гарантирует
    потокобезопасность при работе с не-thread-safe моделью.
    
    Attributes:
        input_queue: Очередь входящих задач от эндпоинтов
        output_queue: Очередь исходящих результатов для диспетчера
        encoder: Загруженная модель SentenceTransformer
        tokenizer: Токенизатор из модели
        running: Флаг работы воркера
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
        """
        Загружает модель и запускает основной цикл обработки задач.
        """
        try:
            model_id = config.HUGGING_FACE_MODEL_NAME
            
            # Скачиваем модель в локальную папку, если её нет
            safe_model_name = model_id.replace('/', '--')
            model_path = config.MODEL_PATH / safe_model_name
            
            if not model_path.exists():
                logger.info(f"Модель не найдена, скачиваю {model_id} в {model_path}")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                    ignore_patterns=[],
                    max_workers=4,
                    resume_download=True
                )
                logger.info(f"Модель успешно скачана в {model_path}")
            else:
                logger.info(f"Модель уже существует в {model_path}")
            
            # Загружаем модель в отдельном потоке, чтобы не блокировать event loop
            self.encoder = await asyncio.to_thread(
                SentenceTransformer,
                str(model_path),
                device=config.DEVICE
            )
            
            # Токенизатор доступен через модель
            self.tokenizer = self.encoder.tokenizer
            
            logger.info(f"Worker: модель загружена, начинаю обработку задач")

            # Основной цикл обработки задач
            while self.running:
                try:
                    # Ждем новую задачу с таймаутом, чтобы проверять флаг running
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
    
    def _build_token_info(self, encodings):
        """
        Преобразует выход токенизатора в формат TEI.
        
        Args:
            encodings: Результат tokenizer() с return_offsets_mapping=True.
                    Это объект BatchEncoding, который ведет себя как словарь со следующими ключами:
                    
                    - input_ids: List[List[int]] 
                        Список списков ID токенов для каждого текста в батче.
                        Это числовое представление, которое модель получает на вход.
                        Пример: [[101, 2054, 2003, 102], [101, 3231, 4242, 102]]
                        
                    - attention_mask: List[List[int]] (опционально)
                        Маска из 1 и 0: 1 — реальный токен, 0 — паддинг (добавлен для выравнивания)
                        
                    - token_type_ids: List[List[int]] (опционально)
                        Для моделей с несколькими предложениями (BERT): 
                        0 — первое предложение, 1 — второе
                        
                    - offset_mapping: List[List[Tuple[int, int]]] (опционально)
                        Позиции токенов в исходном тексте: каждая пара (start, stop) 
                        указывает, где этот токен находится в оригинальной строке.
                        Для специальных токенов ([CLS], [SEP]) обычно (0,0) или None.
                        Пример: [[(0,0), (0,5), (6,11), (0,0)], ...]
        
        Returns:
            List[List[TokenInfo]]: Список списков токенов для каждого входного текста.
        """
        result = []
        mode = config.TOKENIZE_MODE
        
        for text_idx in range(len(encodings['input_ids'])):
            input_ids = encodings['input_ids'][text_idx]
            # tokens() - это метод, вызываем его для каждого индекса
            tokens_text = encodings.tokens(text_idx)
            
            if mode == "full":
                tokens_info = []
                offsets = encodings.get('offset_mapping')
                text_offsets = offsets[text_idx] if offsets else None
                
                for i, token_id in enumerate(input_ids):
                    token_text = tokens_text[i]
                    special = token_text.startswith('[') and token_text.endswith(']')
                    
                    token_info = {
                        "id": token_id,
                        "text": token_text,
                        "special": special
                    }
                    
                    if text_offsets and i < len(text_offsets):
                        start, stop = text_offsets[i]
                        if start != 0 or stop != 0:
                            token_info["start"] = start
                            token_info["stop"] = stop
                    
                    tokens_info.append(token_info)
                result.append(tokens_info)
                
            else:  # mode == "lite"
                tokens_info = [
                    {
                        "id": token_id,
                        "text": "",
                        "special": False,
                    }
                    for token_id in input_ids
                ]
                result.append(tokens_info)
        
        return result
    
    def _normalize_embedding(self, embedding: NDArray[np.float64]) -> NDArray[np.float64]:
        norm = np.linalg.norm(embedding)
        if norm:
            return embedding / norm
        
    def _truncate_texts(self, texts: List[str]) -> List[str]:
        """
        Обрезает тексты до max_seq_length модели, если нужно.
        Возвращает обрезанные тексты.
        """
        if not self.tokenizer or not self.encoder.max_seq_length:
            return texts
        
        max_length = self.encoder.max_seq_length
        result = []
        
        for text in texts:
            # Токенизируем без обрезки, чтобы узнать реальную длину
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= max_length:
                result.append(text)
                continue
            
            # Обрезаем токены (убираем лишние, но сохраняем специальные)
            truncated_tokens = tokens[:max_length]
            
            # Декодируем обратно в текст
            truncated_text = self.tokenizer.decode(
                truncated_tokens, 
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True
            )
            result.append(truncated_text)
            
            logger.debug(f"Текст обрезан с {len(tokens)} до {max_length} токенов")
        
        return result
    
    def _is_too_long(self, text: str) -> bool:
        """
        Проверяет, превышает ли текст лимит модели в токенах.
        Используется для проверки в случае, когда клиент не хочет обрезать тексты.
        """
        if not self.tokenizer or not self.encoder.max_seq_length:
            return False
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens) > self.encoder.max_seq_length

    async def _process_task(self, task: Task):
        """
        Обрабатывает одну задачу.
        
        Это единственное место во всем сервисе, где используется модель.
        """
        start_time = time.time()
        logger.debug(f"Worker: начал обработку задачи {task.task_id} типа {task.task_type}")
        
        try:
            result_data = None
            
            # ===== EMBEDDINGS =====
            if task.task_type == TaskType.ENCODE:
                # Одиночное кодирование
                text = task.data

                # Применяем обрезку если нужно
                if task.truncate:
                    texts = self._truncate_texts([text])
                    text = texts[0]
                # else: проверяем длину и кидаем ошибку если превышает
                elif self._is_too_long(text):
                    raise ValueError(
                        f"Text exceeds max_input_length ({self.encoder.max_seq_length} "
                        "tokens) and truncate=false"
                    )
                
                embedding: NDArray[np.float64] = self.encoder.encode(
                    sentences=text,
                    prompt_name=task.prompt_name
                )
                # Если требуется нормализация
                if task.normalize:
                    embedding = self._normalize_embedding(embedding)
                # TEI требует список списков даже для одного текста
                result_data = [clean_embedding(embedding.tolist())]
                
            elif task.task_type == TaskType.ENCODE_BATCH:
                # Пакетное кодирование с разбивкой по MAX_MODEL_BATCH_SIZE
                texts = task.data

                # Применяем обрезку ко всем текстам
                if task.truncate:
                    texts = self._truncate_texts(texts)
                else:
                    # Проверяем каждый текст
                    for text in texts:
                        if self._is_too_long(text):
                            raise ValueError(
                                f"Text exceeds max_input_length ({self.encoder.max_seq_length}" 
                                "tokens) and truncate=false"
                            )
                
                # Разбиваем на порции по MAX_MODEL_BATCH_SIZE (GPU memory)
                model_batch_size = config.MAX_MODEL_BATCH_SIZE
                all_embeddings = []
                
                for i in range(0, len(texts), model_batch_size):
                    batch = texts[i:i + model_batch_size]
                    logger.debug(
                        f"Worker: порция {i//model_batch_size + 1}/"
                        f"{(len(texts)-1)//model_batch_size + 1} "
                        f"(размер {len(batch)})"
                    )
                    
                    # Кодируем порцию
                    embeddings_batch: NDArray[np.float64] = self.encoder.encode(
                        sentences=batch,
                        prompt_name=task.prompt_name
                    )
                    
                    # Собираем результаты
                    for j in range(embeddings_batch.shape[0]):
                        emb_row: NDArray[np.float64] = embeddings_batch[j]
                        # Нормализация (если затребована)
                        if task.normalize:
                            emb_row = self._normalize_embedding(emb_row)
                        all_embeddings.append(clean_embedding(emb_row.tolist()))
                
                result_data = all_embeddings
            
            # ===== TOKENIZATION =====
            elif task.task_type == TaskType.TOKENIZE:
                
                texts = task.data
                encodings = self.tokenizer(
                    texts,
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                    padding=False,
                    truncation=None
                )
                result_data = self._build_token_info(encodings)
            
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
        """Останавливает воркер и освобождает ресурсы."""
        self.running = False
        logger.info(f"Worker: остановлен. Обработано задач: {self.tasks_processed}")