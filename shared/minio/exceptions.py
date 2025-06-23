"""
Исключения для MinIO API
"""


class MinIOError(Exception):
    """Базовое исключение для MinIO операций"""
    pass


class ObjectNotFoundError(MinIOError):
    """Исключение когда объект не найден в MinIO"""
    pass


class UploadError(MinIOError):
    """Исключение при ошибке загрузки в MinIO"""
    pass


class DownloadError(MinIOError):
    """Исключение при ошибке скачивания из MinIO"""
    pass


class BucketError(MinIOError):
    """Исключение при работе с bucket"""
    pass


class ValidationError(MinIOError):
    """Исключение валидации данных"""
    pass 