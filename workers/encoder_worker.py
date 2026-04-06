# workers/encoder_worker.py

"""
Модуль воркера для Encoder Service.
Содержит класс ModelWorker - единственного владельца модели SentenceTransformer,
"""

import os
# Запрещаем внутреннюю многопоточность токенизаторов, чтобы избежать проблем с fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from shared.config import config
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import asyncio
from typing import List
import math
import numpy as np
from numpy.typing import NDArray
from shared.task import Task, TaskType, TaskResult

from workers.base_worker import BaseWorker

import logging
logger = logging.getLogger(__name__)


def nan_inf_embedding_clean(embedding: List[float]) -> List[float]:
    """
    Очищает вектор от NaN и Inf в соответствии с настройками.
    
    Если EMBEDDING_CLEAN_NAN = false, возвращает исходный вектор (может содержать NaN).
    Если true - заменяет NaN/Inf на EMBEDDING_NAN_REPLACEMENT.
    """
    if not config.EMBEDDING_CLEAN_NAN:
        return embedding
    
    cleaned = []
    nan_idx = []
    inf_idx = []
    
    for i, x in enumerate(embedding):
        if math.isnan(x):
            nan_idx.append(i)
            cleaned.append(config.EMBEDDING_NAN_REPLACEMENT)
        elif math.isinf(x):
            inf_idx.append(i)
            cleaned.append(config.EMBEDDING_NAN_REPLACEMENT)
        else:
            cleaned.append(x)
    
    if config.EMBEDDING_LOG_NAN:
        if nan_idx:
            logger.warning(
                f"{len(nan_idx)} NaN values detected in embedding at indexes: {nan_idx} "
                f"Replaced with {config.EMBEDDING_NAN_REPLACEMENT}"
            )
        if inf_idx:
            logger.warning(
                f"{len(inf_idx)} Inf values detected in embedding at indexes: {inf_idx} "
                f"Replaced with {config.EMBEDDING_NAN_REPLACEMENT}"
            )
    
    return cleaned


class EncoderWorker(BaseWorker):
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
        super().__init__(input_queue, output_queue)
        self.encoder = None
        self.tokenizer = None
        
    async def load_model(self):
        """
        Загружает модель и запускает основной цикл обработки задач.
        """
        try:
            model_id = config.HUGGING_FACE_MODEL_NAME
            
            # Скачиваем модель в локальную папку, если её нет
            safe_model_name = model_id.replace('/', '--')
            model_path = config.SENTENCE_TRANSFORMERS_MODEL_PATH / safe_model_name
            
            if not model_path.exists():
                logger.info(f"Модель не найдена, скачиваю {model_id} в {model_path}")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(model_path),
                    # local_dir_use_symlinks=False, # устаревший параметр
                    ignore_patterns=[],
                    max_workers=4,
                    # resume_download=True          # устаревший параметр
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
            
        except Exception as e:
            logger.error(f"Worker: критическая ошибка при загрузке модели: {e}")
            self.running = False
            raise
    
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

    async def process_task(self, task: Task):
        """
        Обрабатывает одну задачу.
        
        Это единственное место во всем сервисе, где используется модель.
        """
        try:
            result_data = None
            
            # ===== EMBEDDINGS =====        
            if task.task_type == TaskType.ENCODE:
                # Пакетное кодирование с разбивкой по MAX_MODEL_BATCH_SIZE

                # Если передали одиночный текст, делаем из него список из одного элемента
                texts = task.data if isinstance(task.data, list) else [task.data]

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
                        prompt_name=task.prompt_name,
                        show_progress_bar=False
                    )
                    
                    # Собираем результаты
                    for j in range(embeddings_batch.shape[0]):
                        emb_row: NDArray[np.float64] = embeddings_batch[j]
                        # Нормализация (если затребована)
                        if task.normalize:
                            emb_row = self._normalize_embedding(emb_row)
                        all_embeddings.append(nan_inf_embedding_clean(emb_row.tolist()))
                
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
            
            return result_data
            
        except Exception as e:
            logger.error(f"Worker: ошибка при обработке задачи {task.task_id}: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Возвращает информацию о модели для эндпоинта /info."""
        from shared.tei_models import PromptInfo
        
        prompts = []
        if (
            hasattr(self.encoder, "prompts") 
            and self.encoder.prompts 
            and isinstance(self.encoder.prompts, dict)
        ):
            # Будем сообщать только о тех промптах, у которых не пустой текст
            prompts = [
                PromptInfo(name=prompt_name, text=prompt_text)
                for prompt_name, prompt_text in self.encoder.prompts.items()
                if prompt_text
            ]
        
        return {
            "model_id": config.HUGGING_FACE_MODEL_NAME,
            "max_input_length": self.encoder.max_seq_length if self.encoder else None,
            "max_client_batch_size": config.MAX_SERVICE_BATCH_SIZE,
            "prompts": prompts
        }
    
    def is_healthy(self) -> bool:
        """Проверяет, загружена ли модель."""
        return self.encoder is not None