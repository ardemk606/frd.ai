"""
FastAPI зависимости для API слоя
"""
from fastapi import Depends

from shared.repository import (
    DatasetRepository,
    DatasetStatusService,
    get_db_connection
)


def get_dataset_repository(
    db_connection=Depends(get_db_connection)
) -> DatasetRepository:
    """
    FastAPI dependency для получения DatasetRepository
    
    Args:
        db_connection: Подключение к базе данных
        
    Returns:
        Экземпляр DatasetRepository
    """
    return DatasetRepository(db_connection)


def get_dataset_status_service(
    db_connection=Depends(get_db_connection)
) -> DatasetStatusService:
    """
    FastAPI dependency для получения DatasetStatusService
    
    Args:
        db_connection: Подключение к базе данных
        
    Returns:
        Экземпляр DatasetStatusService
    """
    repository = DatasetRepository(db_connection)
    return DatasetStatusService(repository) 