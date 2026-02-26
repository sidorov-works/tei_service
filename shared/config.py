# shared/config.py

from dotenv import load_dotenv
import os
from pathlib import Path
from shared.encoder_models import EncoderModelInfo

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
    EMBEDDING_MODEL = EncoderModelInfo(
        name=os.getenv("HUGGING_FACE_MODEL_NAME", "ai-forever/FRIDA"), 
        vector_size=int(os.getenv("VECTOR_SIZE", "1536")),
        max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512")),
        query_prefix=os.getenv("QUERY_PREFIX", "search_query: "),
        document_prefix=os.getenv("DOCUMENT_PREFIX", "search_document: ")
    )

# Глобальный инстанс конфига для импорта. 
# При необходимости поменять в проекте сразу несколько настроек 
# можно будет сделать config инстансом другого класса
config = DefaultConfig()