# workers/base_worker.py
"""
Базовый класс для всех воркеров TEI Service.

Определяет общий интерфейс и общую логику для:
- EncoderWorker (SentenceTransformer для эмбеддингов)
- ClassifierWorker (AutoModelForSequenceClassification для классификации)

Каждый конкретный воркер реализует свою логику загрузки модели и обработки задач,
но использует общую инфраструктуру очередей, диспетчера и обработки ошибок.
"""

from abc import ABC, abstractmethod
from typing import Any, List
import asyncio
import time
from shared.task import Task, TaskResult
from shared.config import config

import logging
logger = logging.getLogger(__name__)

class BaseWorker(ABC):
    """
    Абстрактный базовый класс для всех воркеров.
    
    Содержит общую логику:
    - Цикл обработки задач из input_queue
    - Отправку результатов в output_queue
    - Очистку текста (единое место для всех типов задач)
    - Замеры времени выполнения
    - Обработку ошибок
    
    Конкретные классы должны реализовать:
    - load_model(): загрузка модели (разная для энкодера и классификатора)
    - process_task(): обработка задачи (разная для энкодера и классификатора)
    - get_model_info(): информация для /info
    - is_healthy(): проверка здоровья
    
    Attributes:
        input_queue: Очередь входящих задач от эндпоинтов
        output_queue: Очередь исходящих результатов для диспетчера
        running: Флаг работы воркера
        tasks_processed: Счетчик обработанных задач
        total_processing_time: Суммарное время обработки всех задач
    """
    
    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        """
        Инициализация воркера.
        
        Args:
            input_queue: Очередь входящих задач от эндпоинтов
            output_queue: Очередь исходящих результатов для диспетчера
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.tasks_processed = 0      # Счетчик обработанных задач
        self.total_processing_time = 0.0  # Суммарное время обработки
        self._clean_text_pattern = config.CLEAN_TEXT_PATTERN
        
    @abstractmethod
    async def load_model(self):
        """
        Загрузка модели.
        
        Для энкодера: SentenceTransformer
        Для классификатора: AutoModelForSequenceClassification
        
        Должна скачивать модель с Hugging Face при необходимости
        и инициализировать self.model, self.tokenizer и т.д.
        """
        pass
    
    @abstractmethod
    async def process_task(self, task: Task) -> Any:
        """
        Обработка задачи.
        
        Args:
            task: Задача для обработки
            
        Returns:
            Any: Результат обработки (формат зависит от типа задачи)
            
        Raises:
            Exception: При любой ошибке обработки
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Информация о модели для эндпоинта /info.
        
        Returns:
            dict с полями:
            - model_id: str
            - max_input_length: Optional[int]
            - max_client_batch_size: int
            - prompts: List[PromptInfo] (для энкодера) или None (для классификатора)
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Проверка здоровья модели.
        
        Returns:
            bool: True если модель загружена и готова к работе
        """
        pass
    
    async def start(self):
        """
        Загружает модель и запускает основной цикл обработки задач.
        
        Этот метод одинаков для всех воркеров:
        1. Загружает модель через load_model()
        2. В бесконечном цикле ждет задачи из input_queue
        3. Для каждой задачи вызывает _process_task() (общая обертка)
        4. Отправляет результат в output_queue
        
        Завершается при установке self.running = False
        """
        try:
            await self.load_model()
            logger.info(f"{self.__class__.__name__}: модель загружена, начинаю обработку задач")
            
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
                    logger.error(f"{self.__class__.__name__}: ошибка в основном цикле: {e}")
                    
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: критическая ошибка при загрузке модели: {e}")
            self.running = False

    def _clean_text(self, text: str) -> str:
        """
        Предварительная очистка текста.
        
        Удаляет нестандартные символы, которые могут вызвать проблемы
        с токенизатором или кодированием.
        """
        if not text:
            return ""
        cleaned_text = self._clean_text_pattern.sub('', text)
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text
    
    async def _process_task(self, task: Task):
        """
        Обрабатывает одну задачу с замерами времени и обработкой ошибок.
        
        Это общая обертка, которая:
        1. Очищает текстовые данные (единое место для всех типов задач)
        2. Замеряет время выполнения
        3. Вызывает специфичную для воркера process_task()
        4. Отправляет результат или ошибку в output_queue
        
        Args:
            task: Задача для обработки
        """
        start_time = time.time()
        logger.debug(f"{self.__class__.__name__}: начал обработку задачи {task.task_id} типа {task.task_type}")
        
        # ОЧИСТКА ТЕКСТА - ЕДИНОЕ МЕСТО 
        # Очищаем все текстовые данные для всех типов задач
        # Это одинаково для энкодера и классификатора
        if isinstance(task.data, str):
            task.data = self._clean_text(task.data)
        elif isinstance(task.data, list):
            task.data = [self._clean_text(t) if isinstance(t, str) else t for t in task.data]
        
        try:
            # Вызываем специфичную для воркера логику обработки
            result_data = await self.process_task(task)
            
            # Отправляем успешный результат
            await self.output_queue.put(TaskResult(
                task_id=task.task_id,
                success=True,
                result=result_data
            ))
            
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            logger.debug(f"{self.__class__.__name__}: завершил задачу {task.task_id} за {processing_time:.3f}с")
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: ошибка при обработке задачи {task.task_id}: {e}")
            await self.output_queue.put(TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e)
            ))
    
    async def stop(self):
        """Останавливает воркер и освобождает ресурсы."""
        self.running = False
        logger.info(f"{self.__class__.__name__}: остановлен. Обработано задач: {self.tasks_processed}")