"""
Pydantic модели для проектов
"""
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class ProjectShortInfo(BaseModel):
    """Краткая информация о проекте"""
    id: int = Field(description="ID проекта")
    name: str = Field(description="Название проекта (название файла)")
    status: str = Field(description="Статус проекта")
    created_at: datetime = Field(description="Дата создания")


class ProjectsShortInfoResponse(BaseModel):
    """Ответ со списком проектов"""
    success: bool
    projects: List[ProjectShortInfo] = Field(description="Список проектов")
    total_count: int = Field(description="Общее количество проектов")


class ProjectDetailInfo(BaseModel):
    """Детальная информация о проекте"""
    id: int = Field(description="ID проекта")
    name: str = Field(description="Название проекта")
    status: str = Field(description="Текущий статус")
    created_at: datetime = Field(description="Дата создания")
    system_prompt: str = Field(description="Системный промпт")
    dataset_preview: List[dict] = Field(description="Первые 5 строк датасета")
    object_name: str = Field(description="Путь к датасету в MinIO")
    system_prompt_object_name: str = Field(description="Путь к промпту в MinIO")


class ProjectDetailResponse(BaseModel):
    """Ответ с детальной информацией о проекте"""
    success: bool
    project: ProjectDetailInfo = Field(description="Детальная информация о проекте") 