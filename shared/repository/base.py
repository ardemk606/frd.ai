"""
Базовые классы для репозиториев
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator
from shared.logging_config import get_logger

logger = get_logger(__name__)


class BaseRepository(ABC):
    """Базовый абстрактный класс для всех репозиториев"""
    
    def __init__(self, db_connection):
        """
        Инициализация репозитория
        
        Args:
            db_connection: Подключение к базе данных
        """
        self._db = db_connection
    
    @contextmanager
    def _get_cursor(self) -> Generator[Any, None, None]:
        """
        Контекстный менеджер для работы с курсором БД
        
        Yields:
            Курсор базы данных
        """
        try:
            with self._db.cursor() as cursor:
                yield cursor
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}", exc_info=True)
            raise
    
    @contextmanager 
    def _get_transaction(self) -> Generator[Any, None, None]:
        """
        Контекстный менеджер для транзакций
        
        Yields:
            Курсор базы данных с автоматическим коммитом/откатом
        """
        try:
            with self._db.cursor() as cursor:
                yield cursor
                self._db.commit()
        except Exception as e:
            self._db.rollback()
            logger.error(f"Ошибка транзакции, выполнен откат: {e}", exc_info=True)
            raise 