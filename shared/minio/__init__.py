"""
MinIO API пакет для работы с объектным хранилищем
"""
from .client import MinIOClient
from .models import (
    ObjectInfo, UploadResult, DownloadResult, ContentType,
    FileUploadRequest, TextUploadRequest, JSONLUploadRequest
)
from .exceptions import MinIOError, ObjectNotFoundError, UploadError, DownloadError
from .dependencies import get_minio_client, create_minio_client_dependency
from .services import DatasetService, PromptService, ProjectStorageService

__all__ = [
    "MinIOClient",
    "ObjectInfo",
    "UploadResult", 
    "DownloadResult",
    "ContentType",
    "FileUploadRequest",
    "TextUploadRequest", 
    "JSONLUploadRequest",
    "MinIOError",
    "ObjectNotFoundError",
    "UploadError",
    "DownloadError",
    "get_minio_client",
    "create_minio_client_dependency",
    "DatasetService",
    "PromptService", 
    "ProjectStorageService"
] 