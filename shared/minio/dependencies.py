"""
Зависимости для инъекции MinIO клиента
"""
from typing import Generator
from functools import lru_cache

from .models import MinIOConfig
from .client import MinIOClient


@lru_cache()
def get_minio_client() -> MinIOClient:
    """
    Получить экземпляр MinIO клиента
    
    Returns:
        Экземпляр MinIOClient
    """
    config = MinIOConfig.from_env()
    return MinIOClient(config)


def create_minio_client_dependency() -> Generator[MinIOClient, None, None]:
    """
    Фабрика для dependency injection в FastAPI
    
    Yields:
        Экземпляр MinIOClient
    """
    client = get_minio_client()
    try:
        yield client
    finally:
        # Здесь можно добавить логику очистки ресурсов
        pass 