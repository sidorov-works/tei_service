# shared/config.py

from dotenv import load_dotenv
import os
from pathlib import Path
from shared.encoder_models import EncoderModelInfo

# load_dotenv берет из текущей папки файл .env 
# и загружает в операционную систему находящиеся в этом файле переменные окружения
load_dotenv(override=True) 

class DefaultConfig:

    ENCODER_NAME = os.getenv("ENCODER_NAME", "default")
    
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")

    EXC_INFO = False # выводить ли в лог весь stacktrace

    LOG_PATH = Path("logs")
    MODEL_PATH = Path("models") / "sentence-transformers"

    # это тоже в коде нужно переделать
    DEVICE = os.getenv("DEVICE", "cpu")  # или "cpu"/"cuda"

    # EMBEDDING_MODEL = EncoderModelInfo(
    #     name="deepvk/USER2-base",
    #     vector_size=768,
    #     max_seq_length=8192,
    #     query_prefix="search_query: ",
    #     document_prefix="search_document: "
    # )

    # Описание и свойства эмбеддинговой модели для кластеризации/классификации обращений
    EMBEDDING_MODEL = EncoderModelInfo(
        name="ai-forever/FRIDA", # название модели на Hugging Face Hub
        vector_size=1536,
        max_seq_length=512,
        query_prefix="search_query: ",
        document_prefix="search_document: "
    )

# Глобальный инстанс конфига для импорта. 
# При необходимости поменять в проекте сразу несколько настроек 
# можно будет сделать config инстансом другого класса
config = DefaultConfig()