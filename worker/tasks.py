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
            - model_id: ID модели для генерации (опционально)
    """
    try:
        project_id = generation_params.get('project_id')
        model_id = generation_params.get('model_id')
        
        logger.info(f"Начинаю генерацию для проекта {project_id}")
        if model_id:
            logger.info(f"Используется модель: {model_id}")
        else:
            logger.info("Используется модель по умолчанию")
        
        # Обновляем статус задачи
        current_task.update_state(
            state='PROGRESS',
            meta={
                'current': 0, 
                'total': generation_params.get('examples_count', 0),
                'model_id': model_id
            }
        )
        
        # Создаем процессор
        processor = DataProcessor()
        
        # Генерируем данные
        result = processor.process_generation_request(generation_params)
        
        logger.info(f"Генерация завершена для проекта {project_id}")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'project_id': project_id,
            'model_id': model_id
        }
        
    except Exception as exc:
        logger.error(f"Ошибка генерации: {exc}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'model_id': generation_params.get('model_id')}
        )
        raise exc


@celery_app.task(bind=True, name='tasks.fine_tune_lora_task')
def fine_tune_lora_task(self, dataset_id: int, output_dir: str = None, model_name: str = None):
    """
    Задача LoRA дообучения с байесовской оптимизацией.
    
    Находит лучшие гиперпараметры, обучает модель и сохраняет адаптер.
    
    Args:
        dataset_id: ID датасета для дообучения.
        output_dir: Директория для сохранения результатов.
        model_name: Название модели для дообучения.
    """
    dataset_path = None
    try:
        # Защита от передачи параметра в виде словаря
        if isinstance(dataset_id, dict):
            dataset_id = dataset_id.get('dataset_id')

        logger.info(f"Начинаю LoRA дообучение для датасета id={dataset_id}")
        self.update_state(state='PROGRESS', meta={'status': 'Получение данных...'})

        # Шаг 1: Получаем имя объекта датасета из БД
        # Используем фабрику для создания репозитория
        from shared.repository.dependencies import create_dataset_repository
        repo = create_dataset_repository()
        dataset_record = repo.get_by_id(dataset_id)
        
        if not dataset_record or not dataset_record.object_name:
            raise FileNotFoundError(f"Запись о датасете {dataset_id} не найдена или не содержит имени файла.")
        
        dataset_object_name = dataset_record.object_name
        logger.info(f"Объект в MinIO для датасета {dataset_id}: {dataset_object_name}")

        # Шаг 2: Скачиваем датасет из MinIO во временный файл
        from shared.minio.dependencies import get_minio_client
        from shared.minio.services import DatasetService
        
        minio_client = get_minio_client()
        dataset_service = DatasetService(minio_client)
        dataset_path = dataset_service.download_dataset_to_temp_file(dataset_object_name)

        # Шаг 3: Запускаем дообучение
        from lora.lora_tuning import LoRATuner, LoRATuningConfig
        
        config = LoRATuningConfig.from_env()
        tuner = LoRATuner(config=config)
        
        final_output_dir = output_dir or f"/app/lora_results/{dataset_id}"
        self.update_state(state='PROGRESS', meta={'status': f'Запуск оптимизации, результаты в {final_output_dir}'})
        
        results = tuner.run_optimization(
            data_path=dataset_path,
            output_dir=final_output_dir,
            model_name=model_name
        )
        
        logger.info(f"LoRA дообучение успешно завершено для датасета {dataset_id}.")
        self.update_state(state='SUCCESS', meta=results)
        
        return results
        
    except Exception as exc:
        logger.error(f"Ошибка LoRA дообучения для датасета {dataset_id}: {exc}", exc_info=True)
        self.update_state(
            state='FAILURE',
            meta={'error_type': type(exc).__name__, 'error_message': str(exc)}
        )
        raise exc
    finally:
        # Шаг 4: Гарантированно удаляем временный файл
        if dataset_path:
            import os
            try:
                os.unlink(dataset_path)
                logger.info(f"Временный файл {dataset_path} удален.")
            except OSError as e:
                logger.error(f"Не удалось удалить временный файл {dataset_path}: {e}")


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