"""
Зависимости для инъекции репозиториев
"""
from functools import lru_cache
from typing import Generator

from .dataset_repository import DatasetRepository
from .status_service import DatasetStatusService


def get_dataset_repository(db_connection) -> Generator[DatasetRepository, None, None]:
    """
    Фабрика для создания DatasetRepository
    
    Args:
        db_connection: Подключение к базе данных
        
    Yields:
        Экземпляр DatasetRepository
    """
    repository = DatasetRepository(db_connection)
    try:
        yield repository
    finally:
        # Здесь можно добавить логику очистки ресурсов
        pass


def get_dataset_status_service(db_connection) -> Generator[DatasetStatusService, None, None]:
    """
    Фабрика для создания DatasetStatusService
    
    Args:
        db_connection: Подключение к базе данных
        
    Yields:
        Экземпляр DatasetStatusService
    """
    repository = DatasetRepository(db_connection)
    service = DatasetStatusService(repository)
    try:
        yield service
    finally:
        # Здесь можно добавить логику очистки ресурсов
        pass 