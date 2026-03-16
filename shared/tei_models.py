# shared/tei_models.py

"""
Pydantic модели для TEI-совместимого API
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Union, List, Optional, Literal
from shared.config import config

class NestedBase(BaseModel):
    model_config = ConfigDict(
        extra='ignore', # незнакомые поля просто игнорируются
        from_attributes=True
    )

class RequestWithInputs(BaseModel):
    """
    Базовая модель для запросов /embed или /tokenize, 
    в которой происходит валидация на ограничения размера батча, 
    длины отдельных текстов и общей длины текстов батча
    """
    inputs: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to embed or tokenize"
    )

    @field_validator('inputs')
    @classmethod
    def validate_inputs(cls, v):
        # Определяем, с чем имеем дело: один текст или список 
        # (в любом случае - приводим к списку)
        texts = [v] if isinstance(v, str) else v
        
        # 1. Проверка на пустой список
        if not texts:
            raise ValueError('Input list cannot be empty')
        
        # 2. Проверка MAX_SERVICE_BATCH_SIZE
        if len(texts) > config.MAX_SERVICE_BATCH_SIZE:
            raise ValueError(
                f'Batch size {len(texts)} exceeds maximum allowed '
                f'{config.MAX_SERVICE_BATCH_SIZE}'
            )
        
        # 3. Проверка MAX_TEXT_LENGTH и MAX_TOTAL_BATCH_LENGTH
        total_length = 0
        for i, text in enumerate(texts):
            text_len = len(text)
            if text_len > config.MAX_TEXT_LENGTH:
                idx_info = f" at index {i}" if isinstance(v, list) else ""
                raise ValueError(
                    f'Text length {text_len}{idx_info} exceeds maximum allowed '
                    f'{config.MAX_TEXT_LENGTH}'
                )
            total_length += text_len
            if total_length > config.MAX_TOTAL_BATCH_LENGTH:
                raise ValueError(
                    f'Total batch length {total_length} exceeds maximum allowed '
                    f'{config.MAX_TOTAL_BATCH_LENGTH}'
                )
        
        return v

# ===========================================================================
# /embed
# ===========================================================================

class EmbedRequest(RequestWithInputs, NestedBase):
    """
    Запрос к /embed endpoint в формате TEI.
    
    Поддерживает как одиночные тексты, так и батчи через поле inputs.

    Наследуем от NestedBase, тем самым допуская, что клиент может передать 
    дополнительные и неизвестные нашему Encoder Service поля, 
    которые будут просто игнорироваться.
    """
    
    # inputs: Union[str, List[str]] 
    # присутствует в родительском классе RequestWithInputs

    prompt_name: Optional[Literal['query', 'document']] = Field(
        None, 
        description="Name of prompt template (e.g., 'query', 'document')"
    )
    normalize: Optional[bool] = Field(
        False, 
        description="Whether to normalize embeddings to unit length"
    )
    truncate: Optional[bool] = Field(
        True, 
        description="Whether to truncate inputs to max_input_length"
    )
    truncation_direction: Optional[Literal['left', 'right']] = Field(
        "right", 
        description="Truncate the right or the left side"
    )


# Ответ TEI не содержит именованных полей - всегда возвращается 
# массив массивов (векторов) - List[List[float]], 
# даже если передавался только один текст)


# ===========================================================================
# /tokenize
# ===========================================================================

class TokenizeRequest(RequestWithInputs, NestedBase):
    """
    Запрос к /tokenize endpoint в формате TEI.

    Поддерживает как одиночные тексты, так и батчи через поле inputs.

    Наследуем от NestedBase, тем самым допуская, что клиент может передать 
    дополнительные и неизвестные нашему Encoder Service поля, 
    которые будут просто игнорироваться.
    """
    # inputs: Union[str, List[str]] 
    # присутствует в родительском классе RequestWithInputs
    
    add_special_tokens: Optional[bool] = Field(
        True, 
        description="Whether to add special tokens to the tokenized output"
    )
    truncate: Optional[bool] = Field(
        True, 
        description="Whether to truncate inputs to max_input_length"
    )

class TokenInfo(NestedBase):
    """
    Информация об одном токене в формате TEI.
    
    Поля start/stop могут отсутствовать (None), если токенизатор
    не поддерживает возврат позиций в исходном тексте.
    """
    id: int = Field(..., description="Token ID in vocabulary")
    text: str = Field(..., description="Token text")
    special: bool = Field(..., description="Whether this is a special token ([CLS], [SEP], etc.)")
    start: Optional[int] = Field(None, description="Start character position in original text")
    stop: Optional[int] = Field(None, description="End character position in original text")

# Результат /tokenize представляет собой массив массивов 
# структур TokenInfo - List[List[TokenInfo]]. 
# При этом сами массивы не являются именованными полями - 
# то есть опять не получается сделать pydantic модель ответа


# ===========================================================================
# /info
# ===========================================================================

# GET запрос к /info не требует параметров

class InfoResponse(NestedBase):
    """
    Ответ от /info endpoint в формате TEI.
    
    Содержит только те поля, которые реально нужны для работы клиента:
    - model_id: идентификация модели
    - max_input_length: ограничение на длину текста
    - max_client_batch_size: ограничение на размер батча
    
    Размерность вектора (dimension) не входит в официальную спецификацию TEI.
    Клиент должен определять её отдельно через тестовый запрос к /embed.
    
    Наследуем от NestedBase, допуская что сервер может вернуть
    дополнительные поля - они будут просто игнорироваться.
    """
    model_id: str = Field(
        ..., 
        description="Hugging Face model ID"
    )
    max_input_length: Optional[int] = Field(
        None, 
        description="Maximum input length in tokens"
    )
    max_client_batch_size: int = Field(
        32,
        description="Maximum number of texts allowed in a single /embed request. "
                    "Client must split larger batches." 
    )