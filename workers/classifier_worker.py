# workers/classifier_worker.py

"""
Модуль воркера для Classifier Service.
Содержит класс ClassifierWorker - наследника BaseWorker,
единственного владельца модели AutoModelForSequenceClassification.

Особенности работы с устройством (CPU/CUDA/MPS):
В отличие от SentenceTransformer (который автоматически управляет устройством
через параметр device в конструкторе), модели из библиотеки transformers
требуют явного переноса как самой модели (model.to(device)), так и входных
тензоров (inputs.to(device)). Это связано с тем, что transformers не
делает предположений о том, на каком устройстве должны выполняться вычисления.
В данном воркере устройство задается через config.DEVICE и может принимать
значения "cpu", "cuda" или "mps" в зависимости от аппаратного обеспечения.
"""

import os
# Запрещаем внутреннюю многопоточность токенизаторов, чтобы избежать проблем с fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from shared.config import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
import asyncio
from typing import List, Any
import torch
import torch.nn.functional as F
from pathlib import Path
from shared.task import Task, TaskType
from workers.base_worker import BaseWorker

import logging
logger = logging.getLogger(__name__)


class ClassifierWorker(BaseWorker):
    """
    Единственный владелец модели классификации.
    
    Все запросы к модели проходят через него последовательно, что гарантирует
    потокобезопасность при работе с не-thread-safe моделью.
    
    Attributes:
        input_queue: Очередь входящих задач от эндпоинтов
        output_queue: Очередь исходящих результатов для диспетчера
        model: Загруженная модель AutoModelForSequenceClassification
        tokenizer: Токенизатор из модели
        device: Устройство (cpu/cuda/mps)
        running: Флаг работы воркера
    """
    
    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        super().__init__(input_queue, output_queue)
        self.model = None
        self.tokenizer = None
        self.device = config.DEVICE
        self.id2label = {}  # Словарь для преобразования ID в метки

    def _get_max_length(self) -> int:
        """
        Определяет максимальную длину последовательности для модели.
        
        Алгоритм:
        1. Пытаемся получить значение из токенизатора (самый надёжный источник)
        2. Если значение некорректное (None, 0, >100000) — ищем в конфиге модели
        3. Если и там нет — fallback на 512
        """
        # Шаг 1: пробуем токенизатор
        raw_length = None
        if hasattr(self.tokenizer, 'model_max_length'):
            raw_length = self.tokenizer.model_max_length
        
        # Шаг 2: проверяем на разумность
        if raw_length and 0 < raw_length <= 100000:
            logger.info(f"Got max_length from tokenizer: {raw_length}")
            return raw_length
        
        # Шаг 3: ищем в конфиге модели
        logger.warning(f"Tokenizer returned insane max_length={raw_length}, checking model config")
        
        if hasattr(self.model.config, 'max_position_embeddings'):
            config_length = self.model.config.max_position_embeddings
            logger.info(f"Using max_position_embeddings from config: {config_length}")
            return config_length
        elif hasattr(self.model.config, 'max_seq_len'):
            config_length = self.model.config.max_seq_len
            logger.info(f"Using max_seq_len from config: {config_length}")
            return config_length
        
        # Шаг 4: fallback
        logger.warning("No valid max_length found in tokenizer or config, using default 512")
        return 512
        
    async def load_model(self):
        """
        Загрузка модели классификации.
        
        Скачивает модель с Hugging Face в локальную папку TRANSFORMERS_MODEL_PATH,
        затем загружает в отдельном потоке (чтобы не блокировать event loop).
        """
        model_id = config.HUGGING_FACE_MODEL_NAME
        
        # Скачиваем модель в локальную папку, если её нет
        safe_model_name = model_id.replace('/', '--')
        model_path = config.TRANSFORMERS_MODEL_PATH / safe_model_name
        
        if not model_path.exists():
            logger.info(f"Модель не найдена, скачиваю {model_id} в {model_path}")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                snapshot_download,
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
        
        # Загружаем токенизатор в отдельном потоке
        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            str(model_path)
        )
        
        # Загружаем модель в отдельном потоке
        self.model = await asyncio.to_thread(
            AutoModelForSequenceClassification.from_pretrained,
            str(model_path)
        )
        
        # Перемещаем модель на нужное устройство
        self.model = self.model.to(self.device)
        self.model.eval()  # Переводим в режим инференса
        
        # Загружаем метки классов, если они есть в конфиге модели
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            self.id2label = self.model.config.id2label
        else:
            # Если меток нет, создаем стандартные LABEL_0, LABEL_1, ...
            num_labels = self.model.config.num_labels
            self.id2label = {str(i): f"LABEL_{i}" for i in range(num_labels)}

        self.max_length = self._get_max_length()        
        logger.info(
            f"ClassifierWorker: модель загружена, num_labels={len(self.id2label)},"
            f"max_length={self.max_length}," 
            f"device={self.device}"
        )
    
    async def process_task(self, task: Task) -> Any:
        """
        Обработка задачи в соответствии с её типом.
        
        Поддерживаемые типы:
        - PREDICT: классификация текстов (одиночный или батч)
        
        Args:
            task: Задача для обработки
            
        Returns:
            Any: Результат обработки (List[List[LabelScore]] для батча)
            
        Raises:
            ValueError: При неподдерживаемом типе задачи
        """
        if task.task_type == TaskType.PREDICT:
            return await self._predict(task)
        else:
            raise ValueError(f"ClassifierWorker не поддерживает тип задачи: {task.task_type}")
    
    async def _predict(self, task: Task) -> List[List[dict]]:
        """
        Классификация текстов.
        
        Этапы:
        1. Приведение входных данных к списку (если пришёл один текст)
        2. Разбивка на порции по MAX_MODEL_BATCH_SIZE
        3. Токенизация каждой порции
        4. Инференс модели
        5. Softmax для получения вероятностей
        6. Преобразование в формат TEI
        """
        # Если передали одиночный текст, делаем из него список из одного элемента
        texts = task.data if isinstance(task.data, list) else [task.data]
        
        model_batch_size = config.MAX_MODEL_BATCH_SIZE
        all_results = []
        
        for i in range(0, len(texts), model_batch_size):
            batch = texts[i:i + model_batch_size]
            logger.debug(
                f"ClassifierWorker: порция {i//model_batch_size + 1}/"
                f"{(len(texts)-1)//model_batch_size + 1} "
                f"(размер {len(batch)})"
            )
            
            # Токенизация порции
            inputs = await asyncio.to_thread(
                self.tokenizer,
                batch,
                return_tensors="pt",
                truncation=task.truncate,
                padding=True,
                max_length=self.max_length
            )
            
            # Переносим на устройство
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Инференс
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    lambda: self.model(**inputs)
                )
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
            
            # Преобразуем результаты
            for j in range(probabilities.shape[0]):
                scores = []
                for k in range(probabilities.shape[1]):
                    label = self.id2label.get(str(k), f"LABEL_{k}")
                    score = probabilities[j][k].item()
                    scores.append({"label": label, "score": score})
                scores.sort(key=lambda x: x["score"], reverse=True)
                all_results.append(scores)
        
        return all_results
    
    def get_model_info(self) -> dict:
        """
        Возвращает информацию о модели для эндпоинта /info.
        
        Формат совместим с TEI.
        
        Returns:
            dict: Информация о модели
        """
        prompts = []
        
        if not self.is_healthy():
            return {
                "model_id": config.HUGGING_FACE_MODEL_NAME,
                "max_input_length": None,
                "max_client_batch_size": config.MAX_SERVICE_BATCH_SIZE,
                "prompts": prompts
            }
        
        return {
            "model_id": config.HUGGING_FACE_MODEL_NAME,
            "max_input_length": self.max_length,
            "max_client_batch_size": config.MAX_SERVICE_BATCH_SIZE,
            "prompts": prompts
        }
    
    def is_healthy(self) -> bool:
        """
        Проверяет, что модель загружена и готова к работе.
        
        Returns:
            bool: True если модель загружена
        """
        return self.model is not None