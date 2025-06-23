"""
Модели данных для MinIO API
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Dict, Any
from io import BytesIO
from enum import Enum


@dataclass
class MinIOConfig:
    """Конфигурация MinIO"""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool = True
    region: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'MinIOConfig':
        """Создать конфигурацию из переменных окружения"""
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            bucket_name=os.getenv("MINIO_BUCKET", "datasets"),
            secure=os.getenv("MINIO_SECURE", "False").lower() == "true",
            region=os.getenv("MINIO_REGION")
        )


class ContentType(Enum):
    """Типы контента для объектов MinIO"""
    TEXT_PLAIN = "text/plain"
    APPLICATION_JSON = "application/json"
    APPLICATION_JSONL = "application/jsonl"
    APPLICATION_OCTET_STREAM = "application/octet-stream"
    TEXT_CSV = "text/csv"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"


@dataclass
class ObjectInfo:
    """Информация об объекте в MinIO"""
    object_name: str
    bucket_name: str
    size: int
    content_type: Optional[str] = None
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None


@dataclass
class UploadResult:
    """Результат загрузки объекта в MinIO"""
    success: bool
    object_info: Optional[ObjectInfo] = None
    error_message: Optional[str] = None
    
    @property
    def object_name(self) -> Optional[str]:
        """Получить имя объекта"""
        return self.object_info.object_name if self.object_info else None


@dataclass
class DownloadResult:
    """Результат скачивания объекта из MinIO"""
    success: bool
    content: Optional[Union[str, bytes]] = None
    object_info: Optional[ObjectInfo] = None
    error_message: Optional[str] = None
    
    @property
    def content_as_text(self) -> Optional[str]:
        """Получить контент как текст"""
        if not self.content:
            return None
        if isinstance(self.content, str):
            return self.content
        return self.content.decode('utf-8')
    
    @property 
    def content_as_bytes(self) -> Optional[bytes]:
        """Получить контент как байты"""
        if not self.content:
            return None
        if isinstance(self.content, bytes):
            return self.content
        return self.content.encode('utf-8')


@dataclass
class FileUploadRequest:
    """Запрос на загрузку файла"""
    object_name: str
    content: Union[str, bytes, BytesIO]
    content_type: Optional[ContentType] = None
    metadata: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Автоопределение content_type если не указан"""
        if self.content_type is None:
            if isinstance(self.content, str):
                self.content_type = ContentType.TEXT_PLAIN
            else:
                self.content_type = ContentType.APPLICATION_OCTET_STREAM


@dataclass
class JSONLUploadRequest:
    """Специальный запрос для загрузки JSONL файлов"""
    object_name: str
    json_objects: list
    metadata: Optional[Dict[str, str]] = None
    
    @property
    def content_type(self) -> ContentType:
        return ContentType.APPLICATION_JSONL


@dataclass
class TextUploadRequest:
    """Специальный запрос для загрузки текстовых файлов"""
    object_name: str
    text_content: str
    metadata: Optional[Dict[str, str]] = None
    
    @property
    def content_type(self) -> ContentType:
        return ContentType.TEXT_PLAIN 