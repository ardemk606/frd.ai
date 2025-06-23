"""
Роутер для валидации датасетов
"""
import logging
from fastapi import APIRouter, HTTPException, Depends

from ..dependencies import get_dataset_repository, get_dataset_status_service
from shared.repository import (
    DatasetRepository,
    DatasetStatusService,
    DatasetNotFoundError,
    RepositoryError,
    ValidationError
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dataset",
    tags=["Validation"],
)


@router.post("/{dataset_id}/validate")
def validate_dataset(
    dataset_id: int,
    repository: DatasetRepository = Depends(get_dataset_repository),
    status_service: DatasetStatusService = Depends(get_dataset_status_service)
):
    """
    Запустить валидацию датасета
    """
    try:
        # Проверяем существование датасета
        dataset = repository.get_by_id(dataset_id)
        logger.info(f"Запуск валидации для датасета {dataset_id}: {dataset.filename}")
        
        # Проверяем что датасет готов к валидации
        if dataset.status != "READY_FOR_VALIDATION":
            raise HTTPException(
                status_code=400,
                detail=f"Датасет не готов к валидации. Текущий статус: {dataset.status}"
            )
        
        # Обновляем статус на VALIDATING
        status_service.set_status(dataset_id, "VALIDATING")
        
        # Получаем Celery app
        from src.routers.projects import get_celery_app
        celery_app = get_celery_app()
        
        # Параметры валидации
        validation_params = {
            'dataset_id': dataset_id,
            'output_type': dataset.output_type,
            'schema': dataset.schema
        }
        
        # Отправляем задачу в cpu_queue (без импорта конкретной задачи)
        task = celery_app.send_task(
            'tasks.validate_dataset_task',
            args=[validation_params],
            queue='cpu_queue'
        )
        
        logger.info(f"Задача валидации отправлена: {task.id} для датасета {dataset_id}")
        
        return {
            "success": True,
            "message": f"Валидация датасета {dataset.filename} запущена",
            "task_id": task.id,
            "queue_name": "cpu_queue",
            "dataset_id": dataset_id
        }
        
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Датасет не найден")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при валидации датасета {dataset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при запуске валидации: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при запуске валидации датасета {dataset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при запуске валидации: {str(e)}"
        ) 