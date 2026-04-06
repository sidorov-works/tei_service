# dispatcher.py

"""
Модуль диспетчера результатов для Encoder Service.
Связывает задачи из очереди с ожидающими их Future и очищает зависшие задачи.
"""

import asyncio
import time
from typing import Dict, Tuple
import concurrent.futures

import logging
logger = logging.getLogger(__name__)


class ResultDispatcher:
    """
    Диспетчер результатов.
    
    Получает результаты от воркера и направляет их в соответствующие Future.
    Периодически очищает задачи, которые висят дольше своего таймаута.
    
    Особенности реализации:
    1. Future.set_result/set_exception выполняются в отдельных потоках через
       ThreadPoolExecutor, чтобы не блокировать event loop колбэками.
    2. Все операции со словарем active_futures защищены блокировкой.
    """
    
    def __init__(self):
        """Инициализация диспетчера."""
        # active_futures: task_id -> (future, created_at, operation_timeout)
        self.active_futures: Dict[str, Tuple[asyncio.Future, float, float]] = {}
        self._lock = asyncio.Lock()
        
        # ThreadPoolExecutor для выполнения set_result/set_exception в потоках
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="future_executor"
        )
        
    def register(self, task_id: str, future: asyncio.Future, operation_timeout: float):
        """
        Регистрирует новую ожидающую задачу.
        
        Args:
            task_id: Уникальный ID задачи
            future: Future для результата
            operation_timeout: Максимальное время выполнения операции
                               (EMBED_TIMEOUT или TOKENIZE_TIMEOUT)
        """
        self.active_futures[task_id] = (future, time.time(), operation_timeout)
        
    def unregister(self, task_id: str):
        """
        Удаляет завершенную задачу.
        
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
                
                future_info = self.active_futures.get(result.task_id)
                if future_info:
                    future, _, _ = future_info
                    
                    if not future.done():
                        if result.success:
                            # set_result может вызывать колбэки, выполняем в потоке
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
    
    async def cleanup_stale(self):
        """
        Периодическая очистка задач, которые превысили свой таймаут.
        
        Проверяет каждые 10 секунд и убивает задачи, которые висят
        в очереди дольше, чем отведено на операцию (EMBED_TIMEOUT или TOKENIZE_TIMEOUT).
        Такие задачи уже не нужны - эндпоинт вернул 504 и Future отменён.
        """
        while True:
            await asyncio.sleep(10)
            now = time.time()
            to_kill = []  # (task_id, future)
            
            # Собираем задачи для убийства под блокировкой
            async with self._lock:
                for task_id, (future, created_at, operation_timeout) in list(self.active_futures.items()):
                    age = now - created_at
                    
                    # Если задача висит дольше своего таймаута и Future ещё не выполнена
                    if age > operation_timeout and not future.done():
                        to_kill.append((task_id, future))
            
            # Убиваем задачи вне блокировки, в отдельных потоках
            for task_id, future in to_kill:
                try:
                    await asyncio.to_thread(
                        self._safe_set_exception,
                        future,
                        asyncio.TimeoutError(
                            f"Task cancelled: operation timeout ({operation_timeout}s) exceeded"
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to kill task {task_id}: {e}")
                
                # Удаляем из словаря (под блокировкой, но быстро)
                async with self._lock:
                    self.active_futures.pop(task_id, None)
            
            if to_kill:
                logger.warning(f"Killed {len(to_kill)} stale tasks (exceeded operation timeout)")
    
    async def get_active_count(self) -> int:
        """Безопасно возвращает количество активных задач."""
        async with self._lock:
            return len(self.active_futures)
    
    async def close(self):
        """Корректное завершение работы."""
        self._executor.shutdown(wait=False)
        logger.debug("Dispatcher closed")