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
        logger.setLevel(logging.DEBUG)
        
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
        logger.addHandler(handler)
        
        # Дублируем в консоль с цветами
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console)
        
        return logger
    except Exception as e:
        logging.error(f"Logger setup failed: {e}")
        return logging.getLogger(name or "root")
    

# Логер для докер-окружения
import logging
from pythonjsonlogger import jsonlogger
import socket
from typing import Optional

def setup_graylog_logger(
    name: Optional[str] = None,
    graylog_host: str = "localhost",
    graylog_port: int = 12201,
    fmt: str = '%(asctime)s %(levelname)s %(processName)s %(message)s'
) -> logging.Logger:
    """
    Логгер для отправки в Graylog/ELK через GELF.
    Требует:
    - pip install python-json-logger pygelf
    """
    try:
        from pygelf import GelfUdpHandler
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Удаляем старые обработчики
        logger.handlers.clear()
        
        # JSON-форматтер
        formatter = jsonlogger.JsonFormatter(fmt)
        
        # Graylog handler
        graylog_handler = GelfUdpHandler(
            host=graylog_host,
            port=graylog_port,
            include_extra_fields=True
        )
        graylog_handler.setFormatter(formatter)
        logger.addHandler(graylog_handler)
        
        # Дублируем в консоль для разработки
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt))
        logger.addHandler(console)
        
        return logger
    except ImportError:
        logging.warning("Graylog dependencies not installed. Falling back to console.")
        return logging.getLogger(name or "root")
    

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Автоматически выбирает оптимальный способ логирования:
    - В продакшене (DOCKER_ENV=True) отправляет в Graylog
    - В разработке использует ConcurrentRotatingFileHandler
    """
    if os.environ.get('DOCKER_ENV') == 'true':
        return setup_graylog_logger(
            name=name,
            graylog_host=os.getenv('GRAYLOG_HOST', 'graylog')
        )
    else:
        return setup_logger(name=name)

logger = get_logger("my_app") # готовый инстанс для импорта


def wrap_logger_methods(logger, worker_name: str):
    """Декоратор для автоматической подстановки process_name"""
    def proc_name_dec(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault('extra', {}).update({'process_name': worker_name})
            return func(*args, **kwargs)
        return wrapper
    
    # Применяем ко всем нужным методам
    logger.info = proc_name_dec(logger.info)
    logger.error = proc_name_dec(logger.error)
    logger.warning = proc_name_dec(logger.warning)
    logger.debug = proc_name_dec(logger.debug)
    logger.critical = proc_name_dec(logger.critical)
    
    return logger