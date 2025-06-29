"""
Подключение к базе данных PostgreSQL
"""
import os
from shared.logging_config import get_logger
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2.extras import RealDictCursor

logger = get_logger(__name__)


class DatabaseConnection:
    """Класс для управления подключением к PostgreSQL"""
    
    def __init__(self):
        """Инициализация подключения к БД"""
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.database = os.getenv("DB_NAME", "datasets")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "password")
        
        self._connection = None
    
    def connect(self):
        """Создать подключение к базе данных"""
        try:
            self._connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                cursor_factory=RealDictCursor
            )
            self._connection.autocommit = False
            logger.info(f"Подключение к PostgreSQL установлено: {self.host}:{self.port}/{self.database}")
            return self._connection
        except Exception as e:
            logger.error(f"Ошибка подключения к PostgreSQL: {e}")
            raise
    
    def disconnect(self):
        """Закрыть подключение к базе данных"""
        if self._connection:
            try:
                self._connection.close()
                logger.info("Подключение к PostgreSQL закрыто")
            except Exception as e:
                logger.error(f"Ошибка при закрытии подключения: {e}")
            finally:
                self._connection = None
    
    def get_connection(self):
        """Получить активное подключение или создать новое"""
        if not self._connection or self._connection.closed:
            return self.connect()
        return self._connection


# Глобальный экземпляр подключения
_db_connection = DatabaseConnection()


@contextmanager
def get_database_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Контекстный менеджер для получения подключения к БД
    
    Yields:
        Подключение к PostgreSQL
    """
    connection = None
    try:
        connection = _db_connection.get_connection()
        yield connection
    except Exception as e:
        logger.error(f"Ошибка работы с БД: {e}")
        if connection:
            connection.rollback()
        raise
    finally:
        # Подключение остается открытым для переиспользования
        pass


def get_db_connection():
    """
    FastAPI dependency для получения подключения к базе данных
    
    Returns:
        Подключение к PostgreSQL
    """
    return _db_connection.get_connection() 