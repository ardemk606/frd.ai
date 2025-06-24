"""
Celery задачи для генерации данных
"""
import logging
from celery import current_task
from celery_app import celery_app
from generate.processor import DataProcessor
from lora.lora_tuning import load_model, fine_tune_lora, prepare_dataset
from lora.lora_tuning_config import LoRATuningConfig

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


@celery_app.task(bind=True, name='tasks.fine_tune_lora_task')
def fine_tune_lora_task(self, fine_tuning_params):
    """
    Задача LoRA дообучения
    
    Args:
        fine_tuning_params: Параметры дообучения
            - dataset_id: ID датасета
            - rank: Rank LoRA [8, 16, 32, 64]
            - lora_alpha: Alpha LoRA (2*rank)
            - learning_rate: Learning rate [1e-5, 5e-4]
            - lora_dropout: Dropout [0.0, 0.15]
    """
    try:
        dataset_id = fine_tuning_params.get('dataset_id')
        logger.info(f"Начинаю LoRA дообучение для датасета {dataset_id}")
        
        # Обновляем статус задачи
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Загрузка модели...'}
        )
        
        # Загружаем конфигурацию
        config = LoRATuningConfig.from_env()
        
        # Подготавливаем параметры LoRA
        lora_params = {
            'rank': fine_tuning_params['rank'],
            'lora_alpha': fine_tuning_params['lora_alpha'], 
            'learning_rate': fine_tuning_params['learning_rate'],
            'lora_dropout': fine_tuning_params['lora_dropout']
        }
        
        # Загружаем модель
        model, tokenizer = load_model()
        
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Подготовка данных...'}
        )
        
        # Загружаем и подготавливаем датасет
        from shared.minio.services import DatasetService
        dataset_service = DatasetService()
        data = dataset_service.get_full_dataset(dataset_id)
        
        dataset = prepare_dataset(data, tokenizer)
        
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 40, 'total': 100, 'status': 'Запуск дообучения...'}
        )
        
        # Запускаем дообучение
        output_dir = f"/tmp/lora_output_{dataset_id}"
        peft_model = fine_tune_lora(model, tokenizer, dataset, lora_params, output_dir)
        
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Сохранение результатов...'}
        )
        
        logger.info(f"LoRA дообучение завершено для датасета {dataset_id}")
        
        return {
            'status': 'SUCCESS',
            'dataset_id': dataset_id,
            'lora_params': lora_params,
            'output_dir': output_dir
        }
        
    except Exception as exc:
        logger.error(f"Ошибка LoRA дообучения: {exc}")
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