"""
Специализированные сервисы для работы с MinIO
"""
from shared.logging_config import get_logger
from datetime import datetime
from typing import List, Dict, Any, Tuple, BinaryIO, Union
from io import BytesIO

from .client import MinIOClient
from .models import (
    FileUploadRequest, TextUploadRequest, JSONLUploadRequest,
    UploadResult, DownloadResult, ContentType
)
from .exceptions import UploadError, DownloadError, ObjectNotFoundError

logger = get_logger(__name__)


class DatasetService:
    """Сервис для работы с датасетами"""
    
    def __init__(self, minio_client: MinIOClient):
        """
        Инициализация сервиса
        
        Args:
            minio_client: Клиент MinIO
        """
        self.client = minio_client
    
    def upload_dataset_file(
        self, 
        file_content: Union[str, bytes, BinaryIO], 
        filename: str,
        timestamp: str = None
    ) -> UploadResult:
        """
        Загружает файл датасета в MinIO
        
        Args:
            file_content: Содержимое файла (строка, байты или файловый объект)
            filename: Имя исходного файла
            timestamp: Временная метка для имени файла
            
        Returns:
            Результат загрузки
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        object_name = f"datasets/{timestamp}_{filename}"
        
        # Создаем запрос на загрузку
        request = FileUploadRequest(
            object_name=object_name,
            content=file_content,
            content_type=ContentType.APPLICATION_JSONL,
            metadata={
                "original_filename": filename,
                "uploaded_at": timestamp,
                "content_type": "application/jsonl"
            }
        )
        
        try:
            result = self.client.upload_file(request)
            logger.info(f"Датасет успешно загружен: {object_name}")
            return result
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета {filename}: {e}")
            raise UploadError(f"Не удалось загрузить датасет: {str(e)}") from e
    
    def get_dataset_preview(self, object_name: str, max_lines: int = 5) -> List[Dict[str, Any]]:
        """
        Получает превью датасета (первые несколько строк)
        
        Args:
            object_name: Имя объекта в MinIO
            max_lines: Максимальное количество строк
            
        Returns:
            Список JSON объектов из датасета
        """
        try:
            preview = self.client.download_jsonl_preview(object_name, max_lines)
            logger.info(f"Получен превью датасета {object_name}: {len(preview)} строк")
            return preview
        except ObjectNotFoundError:
            logger.warning(f"Датасет не найден: {object_name}")
            return [{"error": "Датасет не найден"}]
        except Exception as e:
            logger.error(f"Ошибка получения превью датасета {object_name}: {e}")
            return [{"error": "Не удалось загрузить датасет"}]

    def get_full_dataset(self, object_name: str) -> List[Dict[str, Any]]:
        """
        Загружает полный JSONL датасет
        
        Args:
            object_name: Имя объекта в MinIO
            
        Returns:
            Список всех JSON объектов из датасета
            
        Raises:
            ObjectNotFoundError: Если объект не найден
            DownloadError: При ошибке загрузки
        """
        try:
            import json
            
            # Скачиваем весь файл как текст
            content = self.client.download_text(object_name)
            
            dataset = []
            lines = content.strip().split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    json_obj = json.loads(line)
                    dataset.append(json_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка JSON в строке {line_num} объекта {object_name}: {e}")
                    # Продолжаем обработку, не падаем на одной битой строке
                    
            logger.info(f"Загружен полный датасет {object_name}: {len(dataset)} записей")
            return dataset
            
        except ObjectNotFoundError:
            logger.error(f"Датасет не найден: {object_name}")
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки полного датасета {object_name}: {e}")
            raise DownloadError(f"Не удалось загрузить датасет: {str(e)}") from e

    def download_dataset_to_temp_file(self, object_name: str) -> str:
        """
        Скачивает датасет из MinIO во временный файл и возвращает путь.

        Args:
            object_name: Имя объекта датасета в MinIO.

        Returns:
            Путь к локальному временному файлу с датасетом.
        """
        try:
            logger.info(f"Скачивание датасета {object_name} во временный файл...")
            file_path = self.client.download_object_to_temp_file(object_name)
            logger.info(f"Датасет {object_name} успешно скачан в {file_path}")
            return file_path
        except (ObjectNotFoundError, DownloadError) as e:
            logger.error(f"Не удалось скачать датасет {object_name}: {e}")
            raise


class PromptService:
    """Сервис для работы с системными промптами"""
    
    def __init__(self, minio_client: MinIOClient):
        """
        Инициализация сервиса
        
        Args:
            minio_client: Клиент MinIO
        """
        self.client = minio_client


class ProjectStorageService:
    """Объединенный сервис для работы с файлами проектов"""
    
    def __init__(self, client: MinIOClient):
        self.client = client
        self.dataset_service = DatasetService(client)
        self.prompt_service = PromptService(client)
    
    def get_project_details(
        self, 
        dataset_object_name: str, 
        prompt_object_name: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Получает детали проекта: превью датасета и системный промпт
        
        Args:
            dataset_object_name: Имя объекта датасета в MinIO
            prompt_object_name: Имя объекта промпта в MinIO
            
        Returns:
            Кортеж (превью датасета, системный промпт)
        """
        try:
            # Получаем превью датасета
            dataset_preview = self.dataset_service.get_dataset_preview(dataset_object_name)
            
            # Получаем системный промпт
            system_prompt = self.client.download_text(prompt_object_name)
            
            return dataset_preview, system_prompt
            
        except ObjectNotFoundError as e:
            raise ObjectNotFoundError(f"Файлы проекта не найдены: {e}")
        except Exception as e:
            from .exceptions import MinIOError
            raise MinIOError(f"Ошибка получения данных проекта: {e}")

    def save_generation_result(self, project_id: int, results: List[Dict]) -> str:
        """
        Сохраняет результаты генерации в MinIO
        
        Args:
            project_id: ID проекта
            results: Список результатов генерации
            
        Returns:
            Путь к сохраненному файлу
        """
        try:
            import json
            
            # Генерируем имя файла с timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            object_name = f"project_{project_id}/generated/output_{timestamp}.jsonl"
            
            # Создаем запрос на загрузку JSONL
            jsonl_request = JSONLUploadRequest(
                object_name=object_name,
                json_objects=results,
                metadata={
                    "project_id": str(project_id),
                    "generated_at": timestamp,
                    "type": "generation_result"
                }
            )
            
            upload_result = self.client.upload_jsonl(jsonl_request)
            
            logger.info(f"Результаты генерации сохранены: {object_name}")
            return object_name
            
        except Exception as e:
            raise UploadError(f"Ошибка сохранения результатов генерации для проекта {project_id}: {e}") 