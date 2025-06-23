"""
Роутер для загрузки датасетов
"""
import logging
from datetime import datetime
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form

from ..models import UploadResponse
from shared.repository import (
    DatasetRepository,
    DatasetCreate,
    get_dataset_repository,
    RepositoryError
)
from shared.minio import (
    get_minio_client,
    UploadError,
    MinIOError
)
from ..services.storage import FastAPIStorageService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


@router.post("/dataset", response_model=UploadResponse)
def upload_dataset(
    file: UploadFile = File(...),
    system_prompt: str = Form(...),
    repository: DatasetRepository = Depends(get_dataset_repository)
):
    """
    Загружает датасет и системный промпт в MinIO и записывает метаданные в PostgreSQL
    """
    try:
        # Загружаем файлы проекта через FastAPI сервис
        minio_client = get_minio_client()
        storage_service = FastAPIStorageService(minio_client)
        
        dataset_result, prompt_result = storage_service.upload_project_files(
            dataset_file=file,
            system_prompt=system_prompt
        )
        
        dataset_object_name = dataset_result.object_name
        prompt_object_name = prompt_result.object_name
        dataset_size = dataset_result.object_info.size
        
        # Записываем в PostgreSQL через репозиторий
        dataset_create = DatasetCreate(
            filename=file.filename,
            object_name=dataset_object_name,
            size_bytes=dataset_size,
            system_prompt_object_name=prompt_object_name,
            status='NEW'
        )
        dataset_id = repository.create(dataset_create)
        
        logger.info(f"Загружен проект: датасет {file.filename} как {dataset_object_name}, промпт как {prompt_object_name}, ID={dataset_id}")
        
        return UploadResponse(
            success=True,
            message=f"Проект с датасетом {file.filename} успешно создан",
            object_name=dataset_object_name,
            system_prompt_object_name=prompt_object_name,
            dataset_id=dataset_id
        )
        
    except UploadError as e:
        logger.error(f"Ошибка загрузки файлов в MinIO: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при загрузке файлов: {str(e)}"
        )
    except RepositoryError as e:
        logger.error(f"Ошибка репозитория при загрузке датасета: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при сохранении метаданных: {str(e)}"
        )
    except MinIOError as e:
        logger.error(f"Ошибка MinIO при загрузке датасета: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при работе с файловым хранилищем: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке датасета: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при загрузке: {str(e)}"
        ) 