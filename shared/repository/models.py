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
    json_schema: Optional[str] = None  # JSON схема если output_type = JSON
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
    json_schema: Optional[str] = None  # JSON схема если output_type = JSON


@dataclass
class DatasetUpdate:
    """Модель для обновления датасета"""
    status: Optional[str] = None
    lora_adapter_id: Optional[int] = None
    task_id: Optional[int] = None
    output_type: Optional[str] = None
    json_schema: Optional[str] = None 