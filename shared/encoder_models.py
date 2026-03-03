# shared/encoder_models.py

"""
Pydantic модели для работы с Encoder Service.
Содержит модели для описания энкодера и его свойств.
"""

from pydantic import BaseModel, field_validator, computed_field
from typing import Optional, List
from shared.config import config


# МОДЕЛИ ЗАПРОСОВ С ВАЛИДАЦИЕЙ ===================================================

class EncodeRequest(BaseModel):
    """
    Модель для запроса на кодирование одного текста.
    
    Особенности:
    - Проверяет, что текст не пустой
    - Проверяет, что длина текста не превышает MAX_TEXT_LENGTH
    """
    text: str
    request_type: Optional[str] = "query"  # "query" или "document"
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """
        Валидация одного текста:
        1. Текст не может быть пустым
        2. Длина текста не может превышать MAX_TEXT_LENGTH
        """
        # Проверка на пустой текст
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        
        # Проверка максимальной длины
        if len(v) > config.MAX_TEXT_LENGTH:
            raise ValueError(
                f'Text length ({len(v)} chars) exceeds maximum allowed '
                f'({config.MAX_TEXT_LENGTH} chars)'
            )
        
        return v


class BatchEncodeRequest(BaseModel):
    """
    Модель для запроса на пакетное кодирование нескольких текстов.
    
    Три уровня защиты:
    1. Количество текстов в батче (MAX_BATCH_SIZE)
    2. Длина каждого отдельного текста (MAX_TEXT_LENGTH)
    3. Суммарная длина всех текстов (MAX_BATCH_TOTAL_CHARS)
    """
    texts: List[str]
    request_type: Optional[str] = "query"  # "query" или "document"
    
    @field_validator('texts')
    @classmethod
    def validate_batch_size(cls, v: List[str]) -> List[str]:
        """
        Уровень 1: Проверка количества текстов в батче.
        Не допускает пустые батчи и батчи больше MAX_BATCH_SIZE.
        """
        # Проверка на пустой батч
        if not v:
            raise ValueError('Batch cannot be empty')
        
        # Проверка максимального размера батча
        if len(v) > config.MAX_BATCH_SIZE:
            raise ValueError(
                f'Batch size ({len(v)} texts) exceeds maximum allowed '
                f'({config.MAX_BATCH_SIZE} texts)'
            )
        
        return v
    
    @field_validator('texts')
    @classmethod
    def validate_text_length(cls, v: List[str]) -> List[str]:
        """
        Уровень 2: Проверка длины каждого отдельного текста.
        Ни один текст не должен превышать MAX_TEXT_LENGTH.
        """
        for i, text in enumerate(v):
            if len(text) > config.MAX_TEXT_LENGTH:
                raise ValueError(
                    f'Text at index {i} length ({len(text)} chars) exceeds maximum allowed '
                    f'({config.MAX_TEXT_LENGTH} chars)'
                )
        
        return v
    
    @field_validator('texts')
    @classmethod
    def validate_total_length(cls, v: List[str]) -> List[str]:
        """
        Уровень 3: Проверка суммарной длины всех текстов.
        Предотвращает ситуации, когда много коротких текстов 
        в сумме дают слишком большой объем.
        """
        total_chars = sum(len(text) for text in v)
        
        if total_chars > config.MAX_TOTAL_BATCH_LENGTH:
            raise ValueError(
                f'Total batch length ({total_chars} chars) exceeds maximum allowed '
                f'({config.MAX_TOTAL_BATCH_LENGTH} chars)'
            )
        
        return v


class BatchTokenCountRequest(BaseModel):
    """
    Модель для запроса на пакетный подсчет токенов.
    
    Использует те же лимиты, что и BatchEncodeRequest:
    - MAX_BATCH_SIZE - максимальное количество текстов
    - MAX_TEXT_LENGTH - максимальная длина одного текста
    - MAX_BATCH_TOTAL_CHARS - максимальная суммарная длина
    """
    texts: List[str]
    
    @field_validator('texts')
    @classmethod
    def validate_batch_size(cls, v: List[str]) -> List[str]:
        """Проверка количества текстов в батче"""
        if not v:
            raise ValueError('Batch cannot be empty')
        
        if len(v) > config.MAX_BATCH_SIZE:
            raise ValueError(
                f'Batch size ({len(v)} texts) exceeds maximum allowed '
                f'({config.MAX_BATCH_SIZE} texts)'
            )
        
        return v
    
    @field_validator('texts')
    @classmethod
    def validate_text_length(cls, v: List[str]) -> List[str]:
        """Проверка длины каждого отдельного текста"""
        for i, text in enumerate(v):
            if len(text) > config.MAX_TEXT_LENGTH:
                raise ValueError(
                    f'Text at index {i} length ({len(text)} chars) exceeds maximum allowed '
                    f'({config.MAX_TEXT_LENGTH} chars)'
                )
        
        return v
    
    @field_validator('texts')
    @classmethod
    def validate_total_length(cls, v: List[str]) -> List[str]:
        """Проверка суммарной длины всех текстов"""
        total_chars = sum(len(text) for text in v)
        
        if total_chars > config.MAX_TOTAL_BATCH_LENGTH:
            raise ValueError(
                f'Total batch length ({total_chars} chars) exceeds maximum allowed '
                f'({config.MAX_TOTAL_BATCH_LENGTH} chars)'
            )
        
        return v


# МОДЕЛИ ОТВЕТОВ=====================================================================
    
class BatchTokenCountResponse(BaseModel):
    """
    Модель ответа для пакетного подсчета токенов.
    
    Возвращает список длин в том же порядке, что и входные тексты.
    """
    tokens_counts: List[int]    # количество токенов для каждого текста
    service_available: bool     # статус сервиса
    
    @computed_field
    @property
    def count(self) -> int:     # количество обработанных текстов (для удобства)
        return len(self.tokens_counts)

class EncoderModelInfo(BaseModel):
    """
    Информация о модели, используемой в энкодере.
    
    Содержит все необходимые параметры для работы с конкретной эмбеддинговой моделью.
    Эти данные энкодер отдаёт клиенту при инициализации, чтобы клиент мог
    корректно работать с моделью, не дублируя конфигурацию.
    """
    
    # Название модели на Hugging Face Hub (например: "deepvk/USER2-base")
    name: str
    
    # Размерность выходного вектора (например: 768)
    vector_size: Optional[int] = None
    
    # Максимальная длина текста в токенах (например: 8192 для длинноконтекстных моделей)
    max_seq_length: Optional[int] = None
    
    # Префикс, добавляемый к поисковым запросам (query)
    # Может быть пустой строкой, если модель не требует префиксов
    query_prefix: str = ""
    
    # Префикс, добавляемый к документам при индексации
    # Может быть пустой строкой, если модель не требует префиксов
    document_prefix: str = ""


class EncoderInfo(BaseModel):
    """
    Полное описание экземпляра Encoder Service для клиента.
    
    ВАЖНО: Сервис НЕ знает свой URL - это ответственность клиента.
    URL добавляется клиентом при создании EncoderClient.
    """
    
    # Уникальное имя энкодера (например: "rag", "classifier", "multilingual")
    name: str
    
    # Информация о модели, используемой в этом энкодере
    model_info: EncoderModelInfo
    
    # Статус сервиса (опционально, для информации)
    status: str = "operational"