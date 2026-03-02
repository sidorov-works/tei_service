# shared/config.py

from dotenv import load_dotenv
import os
from pathlib import Path

# load_dotenv берет из текущей папки файл .env 
# и загружает в операционную систему находящиеся в этом файле переменные окружения
load_dotenv(override=True) 

class DefaultConfig:

    ENCODER_NAME = os.getenv("ENCODER_NAME", "frida")
    
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")

    EXC_INFO = False # выводить ли в лог весь stacktrace

    LOG_PATH = Path("logs")
    MODEL_PATH = Path("models") / "sentence-transformers"

    # это тоже в коде нужно переделать
    DEVICE = os.getenv("DEVICE", "cpu")  # или "cpu"/"cuda"

    # Описание и свойства эмбеддинговой модели
    HUGGING_FACE_MODEL_NAME = os.getenv("HUGGING_FACE_MODEL_NAME", "ai-forever/FRIDA")
    QUERY_PREFIX=os.getenv("QUERY_PREFIX", "search_query: ")
    DOCUMENT_PREFIX=os.getenv("DOCUMENT_PREFIX", "search_document: ")

# Глобальный инстанс конфига для импорта. 
# При необходимости поменять в проекте сразу несколько настроек 
# можно будет сделать config инстансом другого класса
config = DefaultConfig()