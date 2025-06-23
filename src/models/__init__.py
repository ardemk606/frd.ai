"""
Главный файл моделей - экспортирует все Pydantic модели для удобства импорта
"""

# Модели генерации
from .generation import (
    GenerateRequest,
    GenerateResponse,
    GenerationRequest,
    GenerationTaskRequest,
    TaskResponse,
)

# Модели загрузки
from .upload import (
    UploadResponse,
)

# Модели проектов
from .projects import (
    ProjectShortInfo,
    ProjectsShortInfoResponse,
    ProjectDetailInfo,
    ProjectDetailResponse,
)

# Task модели больше не нужны

# Системные модели
from .system import (
    HealthResponse,
    InfoResponse
)

# Для удобства импорта из одного места
__all__ = [
    # Generation
    "GenerateRequest",
    "GenerateResponse",
    "GenerationRequest",
    "GenerationTaskRequest",
    "TaskResponse",
    
    # Upload
    "UploadResponse",
    
    # Projects
    "ProjectShortInfo",
    "ProjectsShortInfoResponse",
    "ProjectDetailInfo",
    "ProjectDetailResponse",
    
    # System
    "HealthResponse",
    "InfoResponse",
] 