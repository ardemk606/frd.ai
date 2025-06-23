"""
Исключения для репозиториев
"""


class RepositoryError(Exception):
    """Базовое исключение репозитория"""
    pass


class DatasetNotFoundError(RepositoryError):
    """Исключение когда датасет не найден"""
    pass


class ValidationError(RepositoryError):
    """Исключение валидации данных"""
    pass 