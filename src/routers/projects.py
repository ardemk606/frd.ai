"""
Роутер для работы с проектами
"""
import logging
import json
import os
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from celery import Celery

from ..models.generation import GenerationTaskRequest, TaskResponse

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
            "json_schema": request_body.generation_params.json_schema if request_body.generation_params.is_structured else None
        }
        
        # Отправляем задачу в Celery
        celery_app = get_celery_app()
        task = celery_app.send_task(
            'tasks.generate_dataset_task',
            args=[task_data]
        )
        
        logger.info(f"Задача генерации отправлена для проекта {project_id}, task_id: {task.id}")
        
        return TaskResponse(
            success=True,
            message=f"Задача генерации запущена для проекта {project_id}",
            task_id=task.id,
            queue_name="celery"
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