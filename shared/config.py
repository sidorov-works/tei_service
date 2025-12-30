# shared/config.py

from dotenv import load_dotenv
import os
from pathlib import Path

# load_dotenv берет из текущей папки файл .env 
# и загружает в операционную систему находящиеся в этом файле переменные окружения
load_dotenv(override=True) 

class DefaultConfig:
    
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")
    EXC_INFO = False # выводить ли в лог весь stacktrace

    LOG_PATH = Path("logs")
    MODELS_PATH = Path("models")

    ENCODER_SERVICE_URL = os.getenv("ENCODER_SERVICE_URL")

    ENCODE_TIMEOUT = float(os.getenv("ENCODE_TIMEOUT", "30.0"))
    ENCODE_BATCH_TIMEOUT = float(os.getenv("ENCODE_BATCH_TIMEOUT", "60.0"))

    # Описание и свойства эмбеддинговой модели для RAG
    EMBEDDING_MODEL = {
        "model": "deepvk/USER2-base",  # Название модели на Hugging Face Hub
        "subdir": "sentence-transformers",
        "device": os.getenv("DEVICE", "cpu"),  # или "cpu"/"mps" в зависимости от оборудования
        "vector_size": 768,  # Полная размерность эмбеддингов (Hidden Dim)
        "max_seq_length": 8192,  # Поддерживаемая длина контекста!
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: "
    }

    # Описание и свойства эмбеддинговой модели для кластеризации/классификации обращений
    # EMBEDDING_MODEL = {
    #     "model": "ai-forever/FRIDA",  
    #     "subdir": "sentence-transformers",
    #     "device": os.getenv("DEVICE", "cpu"),  # или "cpu"/"cuda"
    #     "vector_size": 768,
    #     "max_seq_length": 512,  # у FRIDA ограничение 512 токенов ???
    #     "query_prefix": "search_query: ",
    #     "document_prefix": "search_document: "
    # }

# Глобальный инстанс конфига для импорта. 
# При необходимости поменять в проекте сразу несколько настроек 
# можно будет сделать config инстансом другого класса
config = DefaultConfig()