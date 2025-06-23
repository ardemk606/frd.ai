"""
Celery задачи для генерации данных
"""
import logging
from celery import current_task
from celery_app import celery_app
from generate.processor import DataProcessor

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.generate_dataset_task')
def generate_dataset_task(self, generation_params):
    """
    Задача генерации датасета
    
    Args:
        generation_params: Параметры генерации
            - project_id: ID проекта
            - examples_count: Количество примеров
            - is_structured: Структурированные ли данные
            - output_format: Формат вывода
            - json_schema: JSON схема (если нужна)
    """
    try:
        logger.info(f"Начинаю генерацию для проекта {generation_params.get('project_id')}")
        
        # Обновляем статус задачи
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': generation_params.get('examples_count', 0)}
        )
        
        # Создаем процессор
        processor = DataProcessor()
        
        # Генерируем данные
        result = processor.process_generation_request(generation_params)
        
        logger.info(f"Генерация завершена для проекта {generation_params.get('project_id')}")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'project_id': generation_params.get('project_id')
        }
        
    except Exception as exc:
        logger.error(f"Ошибка генерации: {exc}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(exc)}
        )
        raise exc


@celery_app.task(bind=True, name='tasks.validate_dataset_task')
def validate_dataset_task(self, validation_params):
    """
    Задача валидации датасета
    
    Args:
        validation_params: Параметры валидации
            - dataset_id: ID датасета
            - output_type: Тип выходных данных (TEXT/JSON)
            - json_schema: JSON схема для валидации (если есть)
    """
    try:
        logger.info(f"Начинаю валидацию для датасета {validation_params.get('dataset_id')}")
        
        # Обновляем статус задачи
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Загрузка данных...'}
        )
        
        # Создаем валидатор
        from validation.validator import DatasetValidator
        validator = DatasetValidator()
        
        # Запускаем валидацию
        result = validator.validate_dataset(validation_params)
        
        logger.info(f"Валидация завершена для датасета {validation_params.get('dataset_id')}")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'dataset_id': validation_params.get('dataset_id')
        }
        
    except Exception as exc:
        logger.error(f"Ошибка валидации: {exc}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(exc)}
        )
        raise exc 