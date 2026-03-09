# shared/tei_models.py

"""
Pydantic модели для TEI-совместимого API.
Полностью соответствуют спецификации Text Embeddings Inference.
"""

from pydantic import BaseModel, Field, computed_field
from typing import Union, List, Optional

# МОДЕЛИ ЗАПРОСОВ

class EmbedRequest(BaseModel):
    """
    Запрос к /embed endpoint в формате TEI.
    
    Поддерживает как одиночные тексты, так и батчи через поле inputs.
    """
    inputs: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to embed"
    )
    prompt_name: Optional[str] = Field(
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


class TokenizeRequest(BaseModel):
    """
    Запрос к /tokenize endpoint в формате TEI.
    
    Возвращает количество токенов без генерации эмбеддингов.
    """
    inputs: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to tokenize"
    )
    add_special_tokens: Optional[bool] = Field(
        True, 
        description="Whether to add special tokens to the tokenized output"
    )
    truncate: Optional[bool] = Field(
        True, 
        description="Whether to truncate inputs to max_input_length"
    )


class InfoResponse(BaseModel):
    """
    Ответ от /info endpoint в формате TEI.
    
    Содержит метаданные о загруженной модели.
    """
    model_id: str = Field(
        ..., 
        description="Hugging Face model ID or path"
    )
    max_input_length: Optional[int] = Field(
        None, 
        description="Maximum input length in tokens"
    )
    dimension: Optional[int] = Field(
        None, 
        description="Embedding dimension"
    )
    status: str = Field(
        ..., 
        description="Service status: 'operational' or 'degraded'"
    )


# МОДЕЛИ ОТВЕТОВ

class EmbedResponse(BaseModel):
    """
    Ответ от /embed endpoint в формате TEI.
    
    Возвращает либо embedding для одиночного текста,
    либо embeddings для списка текстов.
    """
    embedding: Optional[List[float]] = Field(
        None,
        description="Embedding vector for single input"
    )
    embeddings: Optional[List[List[float]]] = Field(
        None,
        description="List of embedding vectors for batch input"
    )
    
    @computed_field
    @property
    def dimension(self) -> Optional[int]:
        """Размерность вектора (для одиночного запроса)"""
        if self.embedding is not None:
            return len(self.embedding)
        if self.embeddings and len(self.embeddings) > 0:
            return len(self.embeddings[0])
        return None
    
    @computed_field
    @property
    def count(self) -> Optional[int]:
        """Количество обработанных текстов"""
        if self.embeddings is not None:
            return len(self.embeddings)
        if self.embedding is not None:
            return 1
        return None


class TokenizeResponse(BaseModel):
    """
    Ответ от /tokenize endpoint в формате TEI.
    
    Возвращает либо tokens_count для одиночного текста,
    либо tokens_counts для списка текстов.
    """
    tokens_count: Optional[int] = Field(
        None,
        description="Number of tokens for single input"
    )
    tokens_counts: Optional[List[int]] = Field(
        None,
        description="List of token counts for batch input"
    )
    
    @computed_field
    @property
    def count(self) -> Optional[int]:
        """Количество обработанных текстов"""
        if self.tokens_counts is not None:
            return len(self.tokens_counts)
        if self.tokens_count is not None:
            return 1
        return None


class InfoResponse(BaseModel):
    """
    Ответ от /info endpoint в формате TEI.
    
    Содержит метаданные о загруженной модели.
    """
    model_id: str = Field(
        ..., 
        description="Hugging Face model ID or path"
    )
    max_input_length: Optional[int] = Field(
        None, 
        description="Maximum input length in tokens"
    )
    dimension: Optional[int] = Field(
        None, 
        description="Embedding dimension"
    )
    status: str = Field(
        ..., 
        description="Service status: 'operational' or 'degraded'"
    )