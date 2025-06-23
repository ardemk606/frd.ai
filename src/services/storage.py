"""
FastAPI сервисы для работы с файловым хранилищем
"""
from typing import Tuple
from fastapi import UploadFile
from datetime import datetime

from shared.minio import (
    MinIOClient, ProjectStorageService, DatasetService,
    TextUploadRequest, UploadResult
)


class FastAPIStorageService:
    """Сервис для работы с файлами в контексте FastAPI"""
    
    def __init__(self, minio_client: MinIOClient):
        self.client = minio_client
        self.dataset_service = DatasetService(minio_client)
        self.project_storage = ProjectStorageService(minio_client)
    
    def upload_project_files(
        self, 
        dataset_file: UploadFile, 
        system_prompt: str
    ) -> Tuple[UploadResult, UploadResult]:
        """
        Загружает файлы проекта из FastAPI UploadFile
        
        Args:
            dataset_file: Файл датасета из FastAPI
            system_prompt: Системный промпт как строка
            
        Returns:
            Кортеж (результат загрузки датасета, результат загрузки промпта)
        """
        # Читаем содержимое файла
        file_content = dataset_file.file.read()
        
        # Загружаем датасет через shared сервис (без FastAPI зависимостей)
        dataset_result = self.dataset_service.upload_dataset_file(
            file_content=file_content,
            filename=dataset_file.filename
        )
        
        # Загружаем промпт
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_object_name = f"prompts/{timestamp}_system_prompt.txt"
        
        prompt_request = TextUploadRequest(
            object_name=prompt_object_name,
            text_content=system_prompt,
            metadata={
                "uploaded_at": timestamp,
                "type": "system_prompt"
            }
        )
        
        prompt_result = self.client.upload_text(prompt_request)
        
        return dataset_result, prompt_result
    
    def get_project_details(
        self, 
        dataset_object_name: str, 
        prompt_object_name: str
    ) -> Tuple[list, str]:
        """
        Получает детали проекта для FastAPI ответов
        
        Args:
            dataset_object_name: Имя объекта датасета в MinIO
            prompt_object_name: Имя объекта промпта в MinIO
            
        Returns:
            Кортеж (превью датасета, системный промпт)
        """
        return self.project_storage.get_project_details(
            dataset_object_name=dataset_object_name,
            prompt_object_name=prompt_object_name
        ) 