"""
Redis Consumer для чтения задач из Redis и передачи их в Celery
"""
import json
import logging
import time
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import redis
except ImportError:
    redis = None

from celery_app import celery_app
from tasks import generate_dataset_task

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Конфигурация Redis"""
    host: str = "redis"
    port: int = 6379
    db: int = 0
    decode_responses: bool = True
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Создает конфигурацию из переменных окружения"""
        return cls(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0'))
        )


class RedisTaskConsumer:
    """Потребитель задач из Redis"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self._connect()
    
    def _connect(self) -> None:
        """Подключается к Redis"""
        if not redis:
            raise ImportError("Redis не установлен. Установите: pip install redis")
        
        try:
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                decode_responses=self.config.decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Проверяем соединение
            self.redis_client.ping()
            logger.info(f"Подключен к Redis: {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Redis: {e}")
            raise
    
    def consume_tasks(self, queue_name: str = "celery", timeout: int = 1) -> None:
        """
        Потребляет задачи из Redis очереди
        
        Args:
            queue_name: Имя очереди
            timeout: Таймаут блокирующего pop
        """
        logger.info(f"Запуск Redis consumer для очереди {queue_name}")
        
        while True:
            try:
                # Блокирующий pop из очереди
                result = self.redis_client.blpop(queue_name, timeout=timeout)
                
                if result:
                    queue_name_result, message = result
                    logger.info(f"Получено сообщение из очереди {queue_name_result}: {message}")
                    
                    self._process_message(message)
                    
            except redis.RedisError as e:
                logger.error(f"Ошибка Redis: {e}")
                self._reconnect()
                time.sleep(5)
            except KeyboardInterrupt:
                logger.info("Получен сигнал остановки")
                break
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
                time.sleep(5)
    
    def _process_message(self, message: str) -> None:
        """Обрабатывает сообщение из Redis"""
        try:
            # Парсим сообщение
            task_data = json.loads(message)
            
            # Валидируем данные задачи
            if not self._validate_task_data(task_data):
                logger.error(f"Невалидные данные задачи: {task_data}")
                return
            
            # Отправляем задачу в Celery
            generate_dataset_task.delay(task_data)
            logger.info(f"Задача отправлена в Celery: {task_data.get('project_id', 'unknown')}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
        except Exception as e:
            logger.error(f"Ошибка обработки задачи: {e}", exc_info=True)
    
    def _validate_task_data(self, task_data: Dict[str, Any]) -> bool:
        """Валидирует данные задачи"""
        if not isinstance(task_data, dict):
            return False
        
        required_fields = ['project_id']
        for field in required_fields:
            if field not in task_data:
                logger.error(f"Отсутствует обязательное поле: {field}")
                return False
        
        return True
    
    def _reconnect(self) -> None:
        """Переподключается к Redis"""
        logger.info("Попытка переподключения к Redis...")
        try:
            self._connect()
        except Exception as e:
            logger.error(f"Не удалось переподключиться к Redis: {e}")


def main():
    """Основная функция Redis consumer"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        config = RedisConfig.from_env()
        consumer = RedisTaskConsumer(config)
        consumer.consume_tasks()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки. Завершение работы...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 