"""
Shared модуль для работы с репозиториями
"""

from .models import Dataset, DatasetCreate, DatasetUpdate
from .exceptions import DatasetNotFoundError, RepositoryError, ValidationError
from .dataset_repository import DatasetRepository
from .status_service import DatasetStatusService
from .dependencies import create_dataset_repository, create_dataset_status_service
from .database import get_db_connection, get_database_connection

__all__ = [
    # Модели данных
    "Dataset",
    "DatasetCreate", 
    "DatasetUpdate",
    
    # Исключения
    "DatasetNotFoundError",
    "RepositoryError",
    "ValidationError",
    
    # Репозитории
    "DatasetRepository",
    
    # Сервисы
    "DatasetStatusService",
    
    # Фабрики (без FastAPI)
    "create_dataset_repository",
    "create_dataset_status_service",
    
    # База данных
    "get_db_connection",
    "get_database_connection",
] 