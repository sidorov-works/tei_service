# encoder_service/dispatcher.py

"""
Модуль диспетчера результатов для Encoder Service.
Содержит класс ResultDispatcher, который связывает задачи из очереди
с ожидающими их Future.
"""

import asyncio
import time
from typing import Dict, Optional, Tuple, List
import concurrent.futures

from shared.config import config
from shared.utils.logger import logger as base_logger, wrap_logger_methods
from encoder_service.worker import TaskResult

logger = wrap_logger_methods(base_logger, "ENCODER_SERVICE.DISPATCHER")


class ResultDispatcher:
    """
    Диспетчер результатов.
    
    Особенности реализации:
    1. Future.set_exception() всегда вызывается в отдельном потоке через asyncio.to_thread,
       чтобы не блокировать event loop при выполнении колбэков Future.
    2. Все операции со словарем active_futures защищены блокировкой.
    3. cleanup_stale убивает задачи, которые клиент уже не ждет.
    
    Почему to_thread безопасен:
    - Запускает синхронную функцию в отдельном потоке из пула
    - Не блокирует главный event loop
    - Пул потоков ограничен (по умолчанию min(32, os.cpu_count() + 4))
    - set_exception выполняется быстро, даже с колбэками
    """
    
    # Максимальное время жизни задачи без client_timeout
    DEFAULT_MAX_AGE = config.ENCODER_BASE_TIMEOUT + 5
    
    def __init__(self):
        """Инициализация диспетчера."""
        self.active_futures: Dict[str, Tuple[asyncio.Future, float, Optional[float]]] = {}
        self._lock = asyncio.Lock()
        
        # Создаем свой ThreadPoolExecutor для изоляции
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="future_executor"
        )
        
    def register(self, task_id: str, future: asyncio.Future, client_timeout: Optional[float] = None):
        """
        Регистрирует новый ожидающий запрос.
        
        Args:
            task_id: Уникальный ID задачи
            future: Future для результата
            client_timeout: Сколько клиент готов ждать (для cleaner'a)
        """
        self.active_futures[task_id] = (future, time.time(), client_timeout)
        
    def unregister(self, task_id: str):
        """
        Удаляет завершенный запрос.
        
        Args:
            task_id: ID задачи для удаления
        """
        self.active_futures.pop(task_id, None)
        
    async def dispatch(self, result_queue: asyncio.Queue):
        """
        Основной цикл диспетчера.
        
        Получает результаты от воркера и направляет их в Future.
        """
        while True:
            try:
                result = await result_queue.get()
                
                # Ищем Future
                future_info = self.active_futures.get(result.task_id)
                if future_info:
                    future, _, _ = future_info
                    
                    if not future.done():
                        if result.success:
                            # set_result безопасно вызывать в event loop
                            # потому что он не вызывает колбэки синхронно?
                            # НЕТ! set_result ТОЖЕ вызывает колбэки!
                            # Поэтому тоже отправляем в поток
                            await asyncio.to_thread(
                                self._safe_set_result,
                                future,
                                result.result
                            )
                        else:
                            await asyncio.to_thread(
                                self._safe_set_exception,
                                future,
                                Exception(result.error)
                            )
                    
                    # Удаляем из активных
                    self.unregister(result.task_id)
                
                result_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatcher error: {e}")
    
    def _safe_set_result(self, future: asyncio.Future, result):
        """Безопасно устанавливает результат в отдельном потоке."""
        try:
            if not future.done():
                future.set_result(result)
        except Exception:
            pass
    
    def _safe_set_exception(self, future: asyncio.Future, exception: Exception):
        """Безопасно устанавливает исключение в отдельном потоке."""
        try:
            if not future.done():
                future.set_exception(exception)
        except Exception:
            pass
    
    async def cleanup_stale(self, max_service_timeout: int):
        """
        Периодическая очистка задач, которые клиент уже не ждет.
        
        Проверяет каждые 10 секунд и убивает задачи, которые висят
        в очереди дольше, чем клиент готов ждать.
        
        Args:
            max_service_timeout: Жесткий потолок таймаута сервиса
        """
        while True:
            await asyncio.sleep(10)
            now = time.time()
            to_kill = []  # (task_id, future)
            
            # 1. Собираем задачи для убийства под блокировкой
            async with self._lock:
                # Используем list() для безопасной итерации по копии
                for task_id, (future, created_at, client_timeout) in list(self.active_futures.items()):
                    age = now - created_at
                    
                    # Определяем максимальное время жизни задачи
                    if client_timeout:
                        # Клиент явно указал, сколько готов ждать
                        max_age = min(client_timeout, max_service_timeout)
                    else:
                        # Клиент не указал - используем базовый таймаут с запасом
                        max_age = self.DEFAULT_MAX_AGE
                    
                    # Если задача висит дольше допустимого и еще не выполнена
                    if age > max_age and not future.done():
                        to_kill.append((task_id, future))
            
            # 2. Убиваем задачи ВНЕ блокировки, в отдельных потоках
            for task_id, future in to_kill:
                try:
                    # Запускаем set_exception в потоке, чтобы:
                    # - не блокировать event loop колбэками Future
                    # - не держать блокировку на время выполнения колбэков
                    await asyncio.to_thread(
                        self._safe_set_exception,
                        future,
                        asyncio.TimeoutError(
                            f"Task cancelled: client timeout expired after waiting in queue"
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to kill task {task_id}: {e}")
                
                # 3. Удаляем из словаря (под блокировкой, но быстро)
                async with self._lock:
                    self.active_futures.pop(task_id, None)
            
            if to_kill:
                logger.warning(f"Killed {len(to_kill)} stale tasks (clients timed out)")
    
    async def get_active_count(self) -> int:
        """Безопасно возвращает количество активных задач."""
        async with self._lock:
            return len(self.active_futures)
    
    async def close(self):
        """Корректное завершение работы."""
        self._executor.shutdown(wait=False)
        logger.debug("Dispatcher closed")


# Важно! Нужно не забыть закрыть executor при остановке сервиса