"""
Pydantic модели для загрузки датасетов
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UploadResponse(BaseModel):
    """Ответ на загрузку датасета"""
    success: bool
    message: str
    object_name: str = Field(description="Путь к объекту датасета в MinIO")
    system_prompt_object_name: Optional[str] = Field(description="Путь к объекту системного промпта в MinIO")
    dataset_id: int = Field(description="ID записи в PostgreSQL") 