"""
Роутер для системных эндпоинтов (health check, info и т.д.)
"""
from fastapi import APIRouter

from ..models import HealthResponse, InfoResponse

# Создаем роутер для системных эндпоинтов
router = APIRouter(
    tags=["System"],
)


@router.get("/", response_model=InfoResponse)
def read_root():
    """Корневой эндпоинт"""
    return InfoResponse(
        message="Data Generation API", 
        version="1.0.0",
        status="running"
    )


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Проверка здоровья сервиса
    """
    return HealthResponse(
        status="healthy",
        service="data-generation-api"
    ) 