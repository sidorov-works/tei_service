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

    LOG_PATH = Path("logs")
    MODEL_PATH = Path("models") / "sentence-transformers"

    # Настройки для работы с кокретной эмбеддинговой моделью ----------------------------------

    DEVICE = os.getenv("DEVICE", "cpu")  # или "cpu"/"cuda"

    # Описание и свойства эмбеддинговой модели
    HUGGING_FACE_MODEL_NAME = os.getenv("HUGGING_FACE_MODEL_NAME", "ai-forever/FRIDA")
    QUERY_PREFIX=os.getenv("QUERY_PREFIX", "search_query: ")
    DOCUMENT_PREFIX=os.getenv("DOCUMENT_PREFIX", "search_document: ")

    # Лимиты на входящие запросы --------------------------------------------------------------
    # При превышении лимитов сервис должен вызвать ValidationError

    # Максимально допустимое кол-во текстов в батче.
    # Предполагается, что модель Sentence Transformers сама поделит большой батч на порции,
    # однако, для подстраховки от переполнения памяти введем предварительное ограничение
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10000"))

    # Максимально допустимая длина одного текста в запросе. 
    # Этот параметр не связан напрямую со свойством max_seq_len конкретной эмбеддинговой модели 
    # (которая, как ожидается сама обрежет лишнее). Это просто подстраховка, 
    # чтобы не "забить" сервис заведомо выссмысленно огромными запросами
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))

    # Максимально допустимая суммарная длина текстов в батче
    MAX_TOTAL_BATCH_LENGTH = int(os.getenv("MAX_TOTAL_BATCH_LENGTH", "100000"))


    # Настройки логирования и отладки ---------------------------------------------------------
    EXC_INFO = False # выводить ли в лог весь stacktrace

# Глобальный инстанс конфига для импорта. 
# При необходимости поменять в проекте сразу несколько настроек 
# можно будет сделать config инстансом другого класса
config = DefaultConfig()