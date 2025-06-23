"""
Репозиторий для работы с датасетами
"""
import logging
from typing import List, Optional
from datetime import datetime

from .base import BaseRepository
from .models import Dataset, DatasetCreate, DatasetUpdate
from .exceptions import DatasetNotFoundError, RepositoryError

logger = logging.getLogger(__name__)


class DatasetRepository(BaseRepository):
    """Репозиторий для работы с датасетами"""
    
    def get_all(self) -> List[Dataset]:
        """
        Получить все датасеты
        
        Returns:
            Список всех датасетов
            
        Raises:
            RepositoryError: При ошибке выполнения запроса
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT id, filename, status, uploaded_at, object_name, 
                           system_prompt_object_name, size_bytes, lora_adapter_id, task_id
                    FROM datasets
                    ORDER BY uploaded_at DESC
                """)
                
                rows = cursor.fetchall()
                
                datasets = []
                for row in rows:
                    dataset = Dataset(
                        id=row['id'],
                        filename=row['filename'],
                        object_name=row['object_name'],
                        size_bytes=row['size_bytes'],
                        system_prompt_object_name=row['system_prompt_object_name'],
                        status=row['status'],
                        uploaded_at=row['uploaded_at'],
                        lora_adapter_id=row['lora_adapter_id'],
                        task_id=row['task_id']
                    )
                    datasets.append(dataset)
                
                logger.info(f"Получено {len(datasets)} датасетов")
                return datasets
                
        except Exception as e:
            logger.error(f"Ошибка при получении всех датасетов: {e}", exc_info=True)
            raise RepositoryError(f"Не удалось получить список датасетов: {str(e)}") from e
    
    def get_by_id(self, dataset_id: int) -> Dataset:
        """
        Получить датасет по ID
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            Датасет
            
        Raises:
            DatasetNotFoundError: Если датасет не найден
            RepositoryError: При ошибке выполнения запроса
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT id, filename, status, uploaded_at, object_name, 
                           system_prompt_object_name, size_bytes, lora_adapter_id, task_id
                    FROM datasets
                    WHERE id = %s
                """, (dataset_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    raise DatasetNotFoundError(f"Датасет с ID {dataset_id} не найден")
                
                dataset = Dataset(
                    id=row['id'],
                    filename=row['filename'],
                    object_name=row['object_name'],
                    size_bytes=row['size_bytes'],
                    system_prompt_object_name=row['system_prompt_object_name'],
                    status=row['status'],
                    uploaded_at=row['uploaded_at'],
                    lora_adapter_id=row['lora_adapter_id'],
                    task_id=row['task_id']
                )
                
                logger.info(f"Получен датасет с ID {dataset_id}")
                return dataset
                
        except DatasetNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Ошибка при получении датасета {dataset_id}: {e}", exc_info=True)
            raise RepositoryError(f"Не удалось получить датасет: {str(e)}") from e
    
    def create(self, dataset_data: DatasetCreate) -> int:
        """
        Создать новый датасет
        
        Args:
            dataset_data: Данные для создания датасета
            
        Returns:
            ID созданного датасета
            
        Raises:
            RepositoryError: При ошибке выполнения запроса
        """
        try:
            with self._get_transaction() as cursor:
                cursor.execute("""
                    INSERT INTO datasets (filename, object_name, size_bytes, system_prompt_object_name, status, uploaded_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    dataset_data.filename,
                    dataset_data.object_name,
                    dataset_data.size_bytes,
                    dataset_data.system_prompt_object_name,
                    dataset_data.status,
                    datetime.now()
                ))
                
                dataset_id = cursor.fetchone()['id']
                
                logger.info(f"Создан датасет с ID {dataset_id}: {dataset_data.filename}")
                return dataset_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании датасета {dataset_data.filename}: {e}", exc_info=True)
            raise RepositoryError(f"Не удалось создать датасет: {str(e)}") from e
    
    def update_status(self, dataset_id: int, new_status: str) -> None:
        """
        Обновить статус датасета
        
        Args:
            dataset_id: ID датасета
            new_status: Новый статус
            
        Raises:
            DatasetNotFoundError: Если датасет не найден
            RepositoryError: При ошибке выполнения запроса
        """
        try:
            with self._get_transaction() as cursor:
                # Проверяем существование датасета
                cursor.execute("SELECT id FROM datasets WHERE id = %s", (dataset_id,))
                if not cursor.fetchone():
                    raise DatasetNotFoundError(f"Датасет с ID {dataset_id} не найден")
                
                # Обновляем статус
                cursor.execute(
                    "UPDATE datasets SET status = %s WHERE id = %s",
                    (new_status, dataset_id)
                )
                
                logger.info(f"Обновлен статус датасета {dataset_id} на {new_status}")
                
        except DatasetNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Ошибка при обновлении статуса датасета {dataset_id}: {e}", exc_info=True)
            raise RepositoryError(f"Не удалось обновить статус датасета: {str(e)}") from e
    
    def get_status(self, dataset_id: int) -> str:
        """
        Получить статус датасета
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            Статус датасета
            
        Raises:
            DatasetNotFoundError: Если датасет не найден
            RepositoryError: При ошибке выполнения запроса
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT status FROM datasets WHERE id = %s", (dataset_id,))
                row = cursor.fetchone()
                
                if not row:
                    raise DatasetNotFoundError(f"Датасет с ID {dataset_id} не найден")
                
                status = row['status']
                logger.info(f"Получен статус датасета {dataset_id}: {status}")
                return status
                
        except DatasetNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Ошибка при получении статуса датасета {dataset_id}: {e}", exc_info=True)
            raise RepositoryError(f"Не удалось получить статус датасета: {str(e)}") from e
    
    def update(self, dataset_id: int, update_data: DatasetUpdate) -> None:
        """
        Обновить датасет
        
        Args:
            dataset_id: ID датасета
            update_data: Данные для обновления
            
        Raises:
            DatasetNotFoundError: Если датасет не найден
            RepositoryError: При ошибке выполнения запроса
        """
        try:
            # Проверяем что есть что обновлять
            update_fields = []
            update_values = []
            
            if update_data.status is not None:
                update_fields.append("status = %s")
                update_values.append(update_data.status)
            
            if update_data.lora_adapter_id is not None:
                update_fields.append("lora_adapter_id = %s")
                update_values.append(update_data.lora_adapter_id)
                
            if update_data.task_id is not None:
                update_fields.append("task_id = %s")
                update_values.append(update_data.task_id)
            
            if not update_fields:
                logger.warning(f"Нет полей для обновления датасета {dataset_id}")
                return
            
            update_values.append(dataset_id)
            
            with self._get_transaction() as cursor:
                # Проверяем существование датасета
                cursor.execute("SELECT id FROM datasets WHERE id = %s", (dataset_id,))
                if not cursor.fetchone():
                    raise DatasetNotFoundError(f"Датасет с ID {dataset_id} не найден")
                
                # Выполняем обновление
                query = f"UPDATE datasets SET {', '.join(update_fields)} WHERE id = %s"
                cursor.execute(query, update_values)
                
                logger.info(f"Обновлен датасет {dataset_id}: {update_fields}")
                
        except DatasetNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Ошибка при обновлении датасета {dataset_id}: {e}", exc_info=True)
            raise RepositoryError(f"Не удалось обновить датасет: {str(e)}") from e 