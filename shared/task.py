# shared/task.py

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class TaskType(Enum):
    """
    Типы задач, которые может выполнять сервер.
    """
    ENCODE = "encode"       # /embed 
    TOKENIZE = "tokenize"   # /tokenize 
    PREDICT = "predict"     # /predict


@dataclass
class Task:
    """
    Задача для воркера.
    
    Attributes:
        task_id: Уникальный идентификатор задачи (UUID)
        task_type: Тип задачи
        data: Данные для обработки (текст или список текстов)
        created_at: Временная метка создания задачи
        prompt_name: Имя промпта для добавления префикса запроса
        truncate: Обрезать ли тексты длиннее max_input_length
        normalize: Нормализовать ли векторы 
    """
    task_id: str
    task_type: TaskType
    data: Any
    created_at: float
    prompt_name: Optional[str] = None
    truncate: bool = True # только для /embed
    normalize: bool = False # только для /embed


@dataclass
class TaskResult:
    """
    Результат выполнения задачи.
    
    Attributes:
        task_id: ID задачи, к которой относится результат
        success: Флаг успешности выполнения
        result: Результат (если success=True)
        error: Сообщение об ошибке (если success=False)
    """
    task_id: str
    success: bool
    result: Any = None
    error: str = None
