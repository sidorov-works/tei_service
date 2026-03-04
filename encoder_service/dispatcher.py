# encoder_service/dispatcher.py

"""
Модуль диспетчера результатов для Encoder Service.
Содержит класс ResultDispatcher, который связывает задачи из очереди
с ожидающими их Future.
"""

import asyncio
import time
from typing import Dict, Optional, Tuple

from shared.config import config
from shared.utils.logger import logger as base_logger, wrap_logger_methods
from encoder_service.worker import TaskResult

logger = wrap_logger_methods(base_logger, "ENCODER_SERVICE.DISPATCHER")


class ResultDispatcher:
    """
    Диспетчер результатов.
    
    Получает результаты от воркера и направляет их в соответствующие Future.
    Это связующее звено между воркером (который ничего не знает о клиентах)
    и ожидающими запросами.
    
    Как работает:
    1. Эндпоинт регистрирует Future в диспетчере перед отправкой задачи
    2. Воркер кладет результат в очередь результатов
    3. Диспетчер забирает результат, находит по task_id соответствующий Future
    4. Пробуждает ожидающий запрос, устанавливая результат в Future
    
    Attributes:
        active_futures: Словарь task_id -> (future, created_at, client_timeout)
        _lock: Блокировка для безопасного доступа к словарю из нескольких корутин
    """
    
    # Константа для максимального времени жизни задачи без client_timeout
    # Берется из конфига, но с запасом на обработку
    DEFAULT_MAX_AGE = config.ENCODER_BASE_TIMEOUT + 5
    
    def __init__(self):
        """Инициализация диспетчера."""
        # Словарь активных Future: task_id -> (future, created_at, client_timeout)
        # client_timeout нужен для проверки, не пора ли убить задачу
        self.active_futures: Dict[str, Tuple[asyncio.Future, float, Optional[float]]] = {}
        self._lock = asyncio.Lock()
        
    def register(self, task_id: str, future: asyncio.Future, client_timeout: Optional[float] = None):
        """
        Регистрирует новый ожидающий запрос.
        
        Вызывается из эндпоинта перед отправкой задачи воркеру.
        
        Args:
            task_id: Уникальный ID задачи
            future: Future, который будет ждать результат
            client_timeout: Сколько клиент готов ждать (для cleaner'a)
        """
        self.active_futures[task_id] = (future, time.time(), client_timeout)
        
    def unregister(self, task_id: str):
        """
        Удаляет завершенный или отмененный запрос.
        
        Вызывается после того, как результат получен или запрос отменен.
        
        Args:
            task_id: ID задачи для удаления
        """
        self.active_futures.pop(task_id, None)
        
    async def dispatch(self, result_queue: asyncio.Queue):
        """
        Основной цикл диспетчера.
        
        Постоянно слушает очередь результатов от воркера и для каждого результата
        находит соответствующий Future и пробуждает ожидающий запрос.
        
        Args:
            result_queue: Очередь результатов от воркера
        """
        while True:
            try:
                # Ждем результат от воркера
                result = await result_queue.get()
                
                # Ищем Future по task_id
                future_info = self.active_futures.get(result.task_id)
                if future_info:
                    future, _, _ = future_info
                    
                    if result.success:
                        # Успех - пробуждаем запрос с результатом
                        future.set_result(result.result)
                    else:
                        # Ошибка - пробуждаем запрос с исключением
                        future.set_exception(Exception(result.error))
                    
                    # Удаляем из активных
                    self.unregister(result.task_id)
                else:
                    # Future уже нет (клиент отвалился) - просто логируем
                    logger.warning(f"Получен результат для неизвестного task_id: {result.task_id}")
                
                # Отмечаем задачу как выполненную в очереди
                result_queue.task_done()
                
            except asyncio.CancelledError:
                # Корректное завершение при остановке сервиса
                break
            except Exception as e:
                logger.error(f"Dispatcher: ошибка: {e}")
    
    async def cleanup_stale(self, max_service_timeout: int):
        """
        Периодическая очистка "зависших" задач.
        
        Проверяет, не превысили ли задачи максимальное время ожидания.
        Учитывает client_timeout - если клиент просил ждать N секунд,
        а прошло больше, то задачу пора убивать.
        
        Args:
            max_service_timeout: Жесткий потолок таймаута сервиса
        """
        while True:
            await asyncio.sleep(10)  # Проверяем каждые 10 секунд
            now = time.time()
            stale_ids = []
            
            async with self._lock:
                for task_id, (future, created_at, client_timeout) in list(self.active_futures.items()):
                    # Сколько прошло времени с создания задачи
                    age = now - created_at
                    
                    # Определяем максимальное время жизни задачи
                    if client_timeout:
                        # Если клиент указал таймаут - уважаем его (но не больше max_service_timeout)
                        max_age = min(client_timeout, max_service_timeout)
                    else:
                        # Если не указал - используем базовый таймаут с запасом
                        max_age = self.DEFAULT_MAX_AGE
                    
                    if age > max_age and not future.done():
                        stale_ids.append(task_id)
                        # Отменяем Future с таймаутом
                        future.set_exception(
                            asyncio.TimeoutError(f"Task expired after {age:.1f} seconds")
                        )
                
                # Удаляем все зависшие задачи
                for task_id in stale_ids:
                    self.active_futures.pop(task_id, None)
            
            if stale_ids:
                logger.warning(f"Очищено зависших задач: {len(stale_ids)}")