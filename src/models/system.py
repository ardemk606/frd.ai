"""
Pydantic модели для системных эндпоинтов
"""
from pydantic import BaseModel, Field
from typing import Literal


class HealthResponse(BaseModel):
    """Ответ health check эндпоинта"""
    status: Literal["healthy", "unhealthy"] = Field(
        description="Статус здоровья сервиса"
    )
    service: str = Field(description="Название сервиса")


class InfoResponse(BaseModel):
    """Информация о сервисе"""
    message: str = Field(description="Приветственное сообщение")
    version: str = Field(description="Версия API")
    status: Literal["running", "maintenance"] = Field(
        description="Статус работы сервиса"
    ) 