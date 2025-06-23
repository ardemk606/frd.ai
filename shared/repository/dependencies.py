"""
Зависимости для инъекции репозиториев
"""
from .dataset_repository import DatasetRepository
from .status_service import DatasetStatusService
from .database import get_db_connection


def create_dataset_repository() -> DatasetRepository:
    """
    Создать экземпляр DatasetRepository с подключением к БД
    
    Returns:
        Экземпляр DatasetRepository
    """
    db_connection = get_db_connection()
    return DatasetRepository(db_connection)


def create_dataset_status_service() -> DatasetStatusService:
    """
    Создать экземпляр DatasetStatusService с подключением к БД
    
    Returns:
        Экземпляр DatasetStatusService
    """
    db_connection = get_db_connection()
    repository = DatasetRepository(db_connection)
    return DatasetStatusService(repository) 