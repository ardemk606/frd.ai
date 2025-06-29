"""
Общий модуль для настройки JSON-логирования во всех компонентах FRD.ai
Использует python-json-logger для ELK Stack
"""
import logging
import sys
from pythonjsonlogger import jsonlogger
from typing import Optional


def setup_json_logging(service_name: str) -> None:
    """Настройка JSON логирования для всего приложения"""
    
    # Создаем кастомный форматтер с дополнительными полями
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
            
            # Добавляем служебные поля
            log_record['service'] = service_name
            log_record['level'] = record.levelname
            log_record['logger'] = record.name
            log_record['module'] = record.module
            log_record['function'] = record.funcName
            log_record['line'] = record.lineno
            
            # Добавляем Celery-специфичные поля если есть
            if hasattr(record, 'task_name'):
                log_record['task_name'] = record.task_name
            if hasattr(record, 'task_id'):
                log_record['task_id'] = record.task_id
            
            # Процесс
            if hasattr(record, 'processName'):
                log_record['process'] = record.processName
            
            # HTTP-специфичные поля
            if hasattr(record, 'request_id'):
                log_record['request_id'] = record.request_id
            if hasattr(record, 'user_session'):
                log_record['user_session'] = record.user_session
    
    # Настройка root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Удаляем существующие handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Добавляем JSON handler
    json_handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    json_handler.setFormatter(formatter)
    root_logger.addHandler(json_handler)


def get_logger(name: str) -> logging.Logger:
    """Получить logger с JSON форматированием"""
    return logging.getLogger(name) 