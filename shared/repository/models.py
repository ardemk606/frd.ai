"""
Модели данных для репозитория
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Dataset:
    """Модель датасета"""
    id: int
    filename: str
    object_name: str
    size_bytes: int
    system_prompt_object_name: Optional[str]
    status: str
    uploaded_at: datetime
    lora_adapter_id: Optional[int] = None
    task_id: Optional[int] = None


@dataclass
class DatasetCreate:
    """Модель для создания нового датасета"""
    filename: str
    object_name: str
    size_bytes: int
    system_prompt_object_name: Optional[str]
    status: str = "NEW"


@dataclass
class DatasetUpdate:
    """Модель для обновления датасета"""
    status: Optional[str] = None
    lora_adapter_id: Optional[int] = None
    task_id: Optional[int] = None 