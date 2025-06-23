"""
Сервис для управления статусами датасетов
"""
import logging
from typing import List, Tuple

from .dataset_repository import DatasetRepository
from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class DatasetStatusService:
    """Сервис для управления статусами датасетов"""
    
    # Последовательность статусов в пайплайне
    STATUS_PIPELINE = [
        "NEW",
        "GENERATING_DATASET", 
        "READY_FOR_VALIDATION",
        "VALIDATION",
        "READY_FOR_FINE_TUNING",
        "FINE_TUNING",
        "READY_FOR_DEPLOY",
        "DEPLOYED"
    ]
    
    def __init__(self, repository: DatasetRepository):
        """
        Инициализация сервиса
        
        Args:
            repository: Репозиторий для работы с датасетами
        """
        self._repository = repository
    
    def get_next_status(self, current_status: str) -> str:
        """
        Получить следующий статус в пайплайне
        
        Args:
            current_status: Текущий статус
            
        Returns:
            Следующий статус
            
        Raises:
            ValidationError: Если статус неизвестен или это финальный статус
        """
        try:
            current_index = self.STATUS_PIPELINE.index(current_status)
        except ValueError:
            raise ValidationError(f"Неизвестный статус: {current_status}")
        
        if current_index >= len(self.STATUS_PIPELINE) - 1:
            raise ValidationError("Проект уже находится в финальном статусе")
        
        next_status = self.STATUS_PIPELINE[current_index + 1]
        logger.info(f"Следующий статус для {current_status}: {next_status}")
        return next_status
    
    def is_valid_status(self, status: str) -> bool:
        """
        Проверить валидность статуса
        
        Args:
            status: Статус для проверки
            
        Returns:
            True если статус валиден
        """
        return status in self.STATUS_PIPELINE
    
    def get_all_statuses(self) -> List[str]:
        """
        Получить все возможные статусы
        
        Returns:
            Список всех статусов
        """
        return self.STATUS_PIPELINE.copy()
    
    def proceed_to_next_step(self, dataset_id: int) -> Tuple[str, str]:
        """
        Перевести датасет к следующему шагу
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            Кортеж (предыдущий_статус, новый_статус)
            
        Raises:
            DatasetNotFoundError: Если датасет не найден
            ValidationError: Если невозможно перейти к следующему статусу
        """
        # Получаем текущий статус
        current_status = self._repository.get_status(dataset_id)
        
        # Определяем следующий статус
        next_status = self.get_next_status(current_status)
        
        # Обновляем статус
        self._repository.update_status(dataset_id, next_status)
        
        logger.info(f"Датасет {dataset_id} переведён из {current_status} в {next_status}")
        return current_status, next_status
    
    def set_status(self, dataset_id: int, new_status: str) -> None:
        """
        Установить статус датасета с валидацией
        
        Args:
            dataset_id: ID датасета
            new_status: Новый статус
            
        Raises:
            ValidationError: Если статус не валиден
            DatasetNotFoundError: Если датасет не найден
        """
        if not self.is_valid_status(new_status):
            raise ValidationError(f"Недопустимый статус: {new_status}")
        
        self._repository.update_status(dataset_id, new_status)
        logger.info(f"Установлен статус {new_status} для датасета {dataset_id}")
    
    def can_proceed_to_next_step(self, dataset_id: int) -> bool:
        """
        Проверить можно ли перейти к следующему шагу
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            True если можно перейти к следующему шагу
        """
        try:
            current_status = self._repository.get_status(dataset_id)
            self.get_next_status(current_status)
            return True
        except (ValidationError, Exception):
            return False 