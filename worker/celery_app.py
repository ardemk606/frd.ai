"""
Celery приложение для обработки задач генерации данных
"""
import os
from celery import Celery
from shared.logging_config import setup_json_logging, get_logger

setup_json_logging("frd-ai-worker")

# Настройки Celery
celery_app = Celery(
    'data_generation_worker',
    broker=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}",
    backend=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}",
    include=['tasks']
)

# Конфигурация Celery с JSON логированием
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    # Отключаем стандартное форматирование - используем JSON
    worker_hijack_root_logger=False,
    worker_log_format='%(message)s',
    worker_task_log_format='%(message)s',
)

# Логируем старт worker'а
logger = get_logger(__name__)
logger.info("Celery worker application initialized")

if __name__ == '__main__':
    celery_app.start() 