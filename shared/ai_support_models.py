# shared/ai_support_models.py

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from shared.config import config

class EncodeRequest(BaseModel):
    """
    Для передачи энкодеру (через эндпойнт rag-сервиса) текста
    """
    text: str

class BatchEncodeRequest(BaseModel):
    """Запрос на пакетное кодирование"""
    texts: List[str]