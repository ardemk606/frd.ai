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
    output_type: str = "TEXT"  # TEXT или JSON
    schema: Optional[str] = None  # Схема для валидации (JSON, CSV, etc.)
    uploaded_at: datetime = None
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
    output_type: str = "TEXT"  # TEXT или JSON
    schema: Optional[str] = None  # Схема для валидации (JSON, CSV, etc.)


@dataclass
class DatasetUpdate:
    """Модель для обновления датасета"""
    status: Optional[str] = None
    lora_adapter_id: Optional[int] = None
    task_id: Optional[int] = None
    output_type: Optional[str] = None
    schema: Optional[str] = None 