"""
Роутер для работы с проектами
"""
import logging
import json
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from celery import Celery

from ..models.generation import GenerationTaskRequest, TaskResponse, FineTuningTaskRequest

from ..models.projects import (
    ProjectsShortInfoResponse, 
    ProjectShortInfo,
    ProjectDetailResponse,
    ProjectDetailInfo
)
from ..dependencies import get_dataset_repository, get_dataset_status_service
from shared.repository import (
    DatasetRepository, 
    DatasetStatusService, 
    DatasetNotFoundError,
    RepositoryError,
    ValidationError
)
from shared.minio import (
    get_minio_client,
    create_minio_client_dependency,
    ProjectStorageService,
    DatasetService,
    MinIOError,
    ObjectNotFoundError
)
from ..services.storage import FastAPIStorageService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)


@router.get("/short_info", response_model=ProjectsShortInfoResponse)
def get_projects_short_info(
    repository: DatasetRepository = Depends(get_dataset_repository)
):
    """
    Получить краткую информацию о всех проектах
    """
    try:
        datasets = repository.get_all()
        
        projects = []
        for dataset in datasets:
            project = ProjectShortInfo(
                id=dataset.id,
                name=dataset.filename,
                status=dataset.status,
                created_at=dataset.uploaded_at
            )
            projects.append(project)
        
        logger.info(f"Получена информация о {len(projects)} проектах")
        
        return ProjectsShortInfoResponse(
            success=True,
            projects=projects,
            total_count=len(projects)
        )
        
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при получении проектов: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении проектов: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении проектов: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении проектов: {str(e)}"
        )


@router.get("/{project_id}/detail", response_model=ProjectDetailResponse)
def get_project_detail(
    project_id: int,
    repository: DatasetRepository = Depends(get_dataset_repository)
):
    """
    Получить детальную информацию о проекте
    """
    try:
        # Получаем информацию о проекте из репозитория
        logger.info(f"Получаем информацию о проекте {project_id}")
        dataset = repository.get_by_id(project_id)
        
        # Получаем детали проекта через FastAPI сервис
        minio_client = get_minio_client()
        storage_service = FastAPIStorageService(minio_client)
        
        dataset_preview, system_prompt = storage_service.get_project_details(
            dataset_object_name=dataset.object_name,
            prompt_object_name=dataset.system_prompt_object_name
        )
        
        project = ProjectDetailInfo(
            id=dataset.id,
            name=dataset.filename,
            status=dataset.status,
            created_at=dataset.uploaded_at,
            system_prompt=system_prompt,
            dataset_preview=dataset_preview,
            object_name=dataset.object_name,
            system_prompt_object_name=dataset.system_prompt_object_name or ""
        )
        
        logger.info(f"Получена детальная информация о проекте {project_id}")
        
        return ProjectDetailResponse(
            success=True,
            project=project
        )
        
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Проект не найден")
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при получении проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении информации о проекте: {str(e)}"
        )
    except MinIOError as e:
        logger.error(f"Ошибка MinIO при получении файлов проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при загрузке файлов проекта: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении детальной информации о проекте {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении информации о проекте: {str(e)}"
        )


@router.post("/{project_id}/next_step")
def proceed_to_next_step(
    project_id: int,
    status_service: DatasetStatusService = Depends(get_dataset_status_service)
):
    """
    Перейти к следующему шагу пайплайна
    """
    try:
        previous_status, new_status = status_service.proceed_to_next_step(project_id)
        
        logger.info(f"Проект {project_id} переведён из статуса {previous_status} в {new_status}")
        
        return {
            "success": True,
            "message": f"Проект переведён в статус {new_status}",
            "previous_status": previous_status,
            "new_status": new_status
        }
        
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Проект не найден")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при переводе проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при переводе проекта: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при переводе проекта {project_id} к следующему шагу: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при переводе проекта: {str(e)}"
        )


# Celery клиент для отправки задач
def get_celery_app():
    """Получить Celery приложение для отправки задач"""
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_db = os.getenv("REDIS_DB", "0")
    
    celery_app = Celery(
        'data_generation_tasks',
        broker=f'redis://{redis_host}:{redis_port}/{redis_db}',
        backend=f'redis://{redis_host}:{redis_port}/{redis_db}'
    )
    return celery_app


@router.post("/{project_id}/skip_generation")
def skip_generation(
    project_id: int,
    repository: DatasetRepository = Depends(get_dataset_repository),
    status_service: DatasetStatusService = Depends(get_dataset_status_service)
):
    """
    Пропустить генерацию и сразу перейти к валидации на основе seed-датасета
    """
    try:
        # Проверяем что проект существует
        dataset = repository.get_by_id(project_id)
        
        # Проверяем что проект готов к пропуску генерации
        if dataset.status != "NEW":
            raise HTTPException(
                status_code=400,
                detail=f"Пропустить генерацию можно только на статусе NEW. Текущий статус: {dataset.status}"
            )
        
        # Создаем копию seed-датасета в папку generated, чтобы валидация работала корректно
        minio_client = get_minio_client()
        project_storage_service = ProjectStorageService(minio_client)
        
        # Загружаем полный seed-датасет
        dataset_service = DatasetService(minio_client)
        seed_data = dataset_service.get_full_dataset(dataset.object_name)
        
        if not seed_data:
            raise HTTPException(
                status_code=400,
                detail="Seed-датасет пуст или не найден"
            )
        
        # Сохраняем seed-данные как "сгенерированные" (для совместимости с валидацией)
        output_file = project_storage_service.save_generation_result(project_id, seed_data)
        
        # Обновляем статус на READY_FOR_VALIDATION
        status_service.set_status(project_id, "READY_FOR_VALIDATION")
        
        logger.info(f"Генерация пропущена для проекта {project_id}. Seed-датасет скопирован как {output_file}")
        
        return {
            "success": True,
            "message": f"Генерация пропущена. Проект переведён в статус READY_FOR_VALIDATION",
            "output_file": output_file,
            "seed_records_count": len(seed_data)
        }
        
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Проект не найден")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (MinIOError, ObjectNotFoundError) as e:
        logger.error(f"Ошибка MinIO при пропуске генерации для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при работе с файлами: {str(e)}"
        )
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при пропуске генерации для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обновлении статуса: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при пропуске генерации для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при пропуске генерации: {str(e)}"
        )


@router.post("/{project_id}/skip_validation")
def skip_validation(
    project_id: int,
    repository: DatasetRepository = Depends(get_dataset_repository),
    status_service: DatasetStatusService = Depends(get_dataset_status_service)
):
    """
    Пропустить валидацию и сразу перейти к fine-tuning
    """
    try:
        # Проверяем что проект существует
        dataset = repository.get_by_id(project_id)
        
        # Проверяем что проект готов к пропуску валидации
        if dataset.status != "READY_FOR_VALIDATION":
            raise HTTPException(
                status_code=400,
                detail=f"Пропустить валидацию можно только на статусе READY_FOR_VALIDATION. Текущий статус: {dataset.status}"
            )
        
        # Переводим сразу в READY_FOR_FINE_TUNING
        status_service.set_status(project_id, "READY_FOR_FINE_TUNING")
        
        logger.info(f"Валидация пропущена для проекта {project_id}. Статус изменен на READY_FOR_FINE_TUNING")
        
        return {
            "success": True,
            "message": f"Валидация пропущена. Проект переведён в статус READY_FOR_FINE_TUNING",
            "previous_status": "READY_FOR_VALIDATION",
            "new_status": "READY_FOR_FINE_TUNING"
        }
        
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Проект не найден")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при пропуске валидации для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обновлении статуса: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при пропуске валидации для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при пропуске валидации: {str(e)}"
        )


@router.post("/{project_id}/start_generation", response_model=TaskResponse)
def start_generation(
    project_id: int,
    request_body: GenerationTaskRequest,
    repository: DatasetRepository = Depends(get_dataset_repository)
):
    """
    Запустить генерацию датасета для проекта
    """
    try:
        # Проверяем что проект существует
        dataset = repository.get_by_id(project_id)
        
        # Создаем сервис для управления статусами
        status_service = DatasetStatusService(repository)
        
        # Обновляем статус на GENERATING_DATASET
        status_service.set_status(project_id, "GENERATING_DATASET")
        
        # Подготавливаем данные для задачи
        task_data = {
            "project_id": project_id,
            "examples_count": request_body.generation_params.examples_count,
            "is_structured": request_body.generation_params.is_structured,
            "output_format": request_body.generation_params.output_format,
            "json_schema": request_body.generation_params.json_schema if request_body.generation_params.is_structured else None,
            "model_id": request_body.generation_params.model_id  # Добавляем model_id
        }
        
        # Отправляем задачу в Celery
        celery_app = get_celery_app()
        task = celery_app.send_task(
            'tasks.generate_dataset_task',
            args=[task_data]
        )
        
        logger.info(f"Задача генерации отправлена для проекта {project_id}, task_id: {task.id}")
        if request_body.generation_params.model_id:
            logger.info(f"Используется модель: {request_body.generation_params.model_id}")
        
        return TaskResponse(
            success=True,
            message=f"Задача генерации запущена для проекта {project_id}",
            task_id=task.id,
            queue_name="celery",
            model_id=request_body.generation_params.model_id
        )
        
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Проект не найден")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при запуске генерации для проекта {project_id}: {e}", exc_info=True)
        
        # Возвращаем статус обратно в случае ошибки
        try:
            status_service.set_status(project_id, "NEW")
        except Exception as rollback_error:
            logger.error(f"Ошибка при откате статуса проекта: {rollback_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при запуске генерации: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при запуске генерации для проекта {project_id}: {e}", exc_info=True)
        
        # Возвращаем статус обратно в случае ошибки
        try:
            status_service.set_status(project_id, "NEW")
        except Exception as rollback_error:
            logger.error(f"Ошибка при откате статуса проекта: {rollback_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при запуске генерации: {str(e)}"
        )


@router.post("/{project_id}/start_fine_tuning")
async def start_fine_tuning(
    project_id: int,
    request_body: Optional[FineTuningTaskRequest] = None,
    dataset_repository: DatasetRepository = Depends(get_dataset_repository),
    status_service: DatasetStatusService = Depends(get_dataset_status_service)
):
    """Запустить дообучение LoRA для проекта"""
    try:
        # 1. Проверяем, что проект существует и готов к дообучению
        project = dataset_repository.get_by_id(project_id)
        if project.status != "READY_FOR_FINE_TUNING":
            raise HTTPException(
                status_code=400,
                detail=f"Проект находится в статусе {project.status}, а не READY_FOR_FINE_TUNING"
            )

        # 2. Обновляем статус
        status_service.set_status(project_id, "FINE_TUNING")

        # 3. Подготавливаем параметры для задачи
        # Если запрос не передан, используем настройки по умолчанию
        if request_body:
            fine_tuning_params = request_body.fine_tuning_params
        else:
            # Создаем параметры по умолчанию для обратной совместимости
            from ..models.generation import FineTuningRequest
            fine_tuning_params = FineTuningRequest()

        task_data = {
            "dataset_id": project_id,
            "use_llm_judge": fine_tuning_params.use_llm_judge,
            "judge_model_id": fine_tuning_params.judge_model_id,
            "base_model_name": fine_tuning_params.base_model_name,
            "n_trials": fine_tuning_params.n_trials,
            "enable_mlflow": fine_tuning_params.enable_mlflow if fine_tuning_params.enable_mlflow is not None else False
        }

        # 4. Отправляем задачу в очередь GPU
        celery_app = get_celery_app()
        task = celery_app.send_task(
            'tasks.fine_tune_lora_task',
            kwargs=task_data,
            queue='gpu_queue'
        )

        logger.info(f"Задача дообучения LoRA отправлена для проекта {project_id}, task_id: {task.id}")
        logger.info(f"Параметры fine-tuning: LLM Judge={'включен' if fine_tuning_params.use_llm_judge else 'выключен'}, "
                    f"judge_model={fine_tuning_params.judge_model_id or 'по умолчанию'}, "
                    f"base_model={fine_tuning_params.base_model_name or 'по умолчанию'}, "
                    f"n_trials={fine_tuning_params.n_trials}")

        return TaskResponse(
            success=True,
            message=f"Задача дообучения запущена для проекта {project_id}",
            task_id=task.id,
            queue_name="gpu_queue"
        )

    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Проект не найден")
    except RepositoryError as e:
        logger.error(f"Ошибка при запуске дообучения для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера") 