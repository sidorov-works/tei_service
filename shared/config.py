from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(override=True)

class DefaultConfig:

    ENCODER_NAME = os.getenv("ENCODER_NAME", "frida")

    MODEL_PATH = Path("models") / "sentence-transformers"

    # Логирование -----------------------------------------------------------------------------
    LOG_PATH = Path(os.getenv("LOG_PATH", "logs"))
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s')
    DOCKER_ENV= os.getenv("DOCKER_ENV") == "true" 

    # Безопасность и аутентификация -----------------------------------------------------------
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")
    ALLOWED_JWT_ALGORITHMS = ["HS256"]
    REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"

    # Тайм-ауты -------------------------------------------------------------------------------
    # Максимальное время выполнения операции /embed (от входа в эндпоинт до возврата результата)
    EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "30.0"))
    # Максимальное время выполнения операции /tokenize (токенизация выполняется быстрее)
    TOKENIZE_TIMEOUT = float(os.getenv("TOKENIZE_TIMEOUT", "5.0"))

    # Ограничения длин очередей ---------------------------------------------------------------
    INPUT_QUEUE_MAXSIZE = int(os.getenv("INPUT_QUEUE_MAXSIZE", "1000"))
    OUTPUT_QUEUE_MAXSIZE = int(os.getenv("OUTPUT_QUEUE_MAXSIZE", "1000"))
    HEALTH_QUEUE_THRESHOLD = 0.9  # Порог переполненности для /health

    # Настройки для работы с конкретной эмбеддинговой моделью ---------------------------------
    DEVICE = os.getenv("DEVICE", "cpu")  # или "cuda"

    # Название (идентификатор) модели на hugging face
    HUGGING_FACE_MODEL_NAME = os.getenv("HUGGING_FACE_MODEL_NAME", "ai-forever/FRIDA")

    # Обработка NaN/Inf в эмбеддингах ---------------------------------------------------------
    EMBEDDING_CLEAN_NAN = os.getenv("EMBEDDING_CLEAN_NAN", "true").lower() == "true"
    EMBEDDING_NAN_REPLACEMENT = float(os.getenv("EMBEDDING_NAN_REPLACEMENT", "0.0"))
    EMBEDDING_LOG_NAN = os.getenv("EMBEDDING_LOG_NAN", "true").lower() == "true"

    # Режим применения эндпойнта /tokenize ---------------------------------------------------- 
    # Оригинальный TEI возвращает полную информацию о токенах. 
    # Это не всегда и не всем требуется. Например, для основного применения - подсчета 
    # длины текста в токенах - достаточно только определить длину списков с информацией о токенах, 
    # а сама информация не нужна. Поэтому предусматриваем режим "lite", 
    # в котором ненужные поля по токенам будет содержать пустые данные
    TOKENIZE_MODE = os.getenv("TOKENIZE_MODE", "full")

    # Лимиты на входящие запросы --------------------------------------------------------------
    # Максимально допустимое кол-во текстов в батче ДЛЯ ЭНКОДЕРА.
    MAX_MODEL_BATCH_SIZE = int(os.getenv("MAX_MODEL_BATCH_SIZE", "32"))

    # Максимально допустимое кол-во текстов в батче ДЛЯ ЭНДПОЙНТА СЕРВИСА.
    MAX_SERVICE_BATCH_SIZE = int(os.getenv("MAX_SERVICE_BATCH_SIZE", "128"))

    # Максимально допустимая длина одного текста в запросе. 
    # Это ограничительная защита самого сервиса, не связано с max_seq_length модели
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))

    # Максимально допустимая суммарная длина текстов в батче
    # Это ограничительная защита самого сервиса, не связано с max_seq_length модели
    MAX_TOTAL_BATCH_LENGTH = int(os.getenv("MAX_TOTAL_BATCH_LENGTH", "500000"))

    # Rate limiting и защита от Ddos ---------------------------------------------------------
    RATE_LIMIT_INFO = os.getenv("RATE_LIMIT_INFO", "500/minute")
    RATE_LIMIT_HEALTH = os.getenv("RATE_LIMIT_HEALTH", "500/minute")                
    RATE_LIMIT_EMBED = os.getenv("RATE_LIMIT_EMBED", "200/minute")            
    RATE_LIMIT_COUNT_TOKENS = os.getenv("RATE_LIMIT_COUNT_TOKENS", "200/minute")      
    RATE_LIMIT_COUNT_TOKENS_BATCH = os.getenv("RATE_LIMIT_COUNT_TOKENS_BATCH", "60/minute")

config = DefaultConfig()