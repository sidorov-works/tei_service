# shared/utils/logger.py

"""
Кастомный логгер с поддержкой:
- Многопроцессной записи в файл (локальная разработка)
- Docker-окружения (только stdout)
- Управления уровнем логирования через config.LOGGING_LEVEL
- Автоматической подстановки process_name через декоратор
"""

import logging
import os
from concurrent_log_handler import ConcurrentRotatingFileHandler
from typing import Optional
from functools import wraps
from pathlib import Path
from shared.config import config


# Логер для записи в файл (без докера)
def setup_logger(
    name: Optional[str] = None,
    log_file: str = Path(config.LOG_PATH) / "app.log",
    level: int = logging.DEBUG,  # Добавлен параметр level
    fmt: str = '%(asctime)s | %(levelname)-8s | %(process_name)-12s | %(message)s',
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """
    Безопасный логгер для многопроцессной среды.
    Особенности:
    - Использует блокировки для безопасной записи
    - Добавляет имя процесса в логи
    - Поддерживает ротацию
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(level)  # Используем переданный уровень
        
        # Удаляем старые обработчики
        logger.handlers.clear()
        
        # Потокобезопасный ротирующий обработчик
        handler = ConcurrentRotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
            use_gzip=True
        )
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        handler.setLevel(level)  # Устанавливаем уровень для хендлера
        logger.addHandler(handler)
        
        # Дублируем в консоль с цветами
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(message)s'))
        console.setLevel(level)  # Устанавливаем уровень для консоли
        logger.addHandler(console)
        
        return logger
    except Exception as e:
        logging.error(f"Logger setup failed: {e}")
        return logging.getLogger(name or "root")
    

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Автоматически выбирает оптимальный способ логирования:
    - В продакшене (DOCKER_ENV=True) пишет только в stdout, Docker сам собирает
    - В разработке использует ConcurrentRotatingFileHandler + консоль
    """
    log_level = getattr(config, 'LOGGING_LEVEL', 'DEBUG')
    numeric_level = getattr(logging, log_level.upper() if isinstance(log_level, str) else 'DEBUG', logging.DEBUG)
    
    if os.environ.get('DOCKER_ENV') == 'true':
        # В Docker - только stdout, без файлов
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        logger.handlers.clear()
        
        # Единственный handler - stdout
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(process_name)-12s | %(message)s'
        ))
        console.setLevel(numeric_level)
        logger.addHandler(console)
        
        return logger
    else:
        # Вне Docker - пишем в файлы как раньше
        return setup_logger(
            name=name,
            level=numeric_level
        )

# Создаем готовый инстанс для импорта
logger = get_logger("my_app")


def wrap_logger_methods(logger, worker_name: str):
    """Декоратор для автоматической подстановки process_name в extra-поля"""
    def proc_name_dec(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Безопасно добавляем process_name в extra, сохраняя существующие поля
            extra = kwargs.get('extra', {})
            extra['process_name'] = worker_name
            kwargs['extra'] = extra
            return func(*args, **kwargs)
        return wrapper
    
    # Применяем ко всем методам логирования
    for method_name in ['debug', 'info', 'warning', 'error', 'critical']:
        method = getattr(logger, method_name)
        setattr(logger, method_name, proc_name_dec(method))
    
    return logger