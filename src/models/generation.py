"""
Модели для генерации данных
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class GenerationRequest(BaseModel):
    """Запрос на генерацию данных"""
    examples_count: int = Field(ge=1, le=1000, description="Количество примеров для генерации")
    is_structured: bool = Field(description="Является ли ожидаемый ответ структурированным")
    output_format: str = Field(default="json", description="Формат вывода (пока только json)")
    json_schema: Optional[str] = Field(description="JSON-схема ожидаемого результата")
    model_id: Optional[str] = Field(
        default=None, 
        description="ID модели для генерации (если не указан, используется модель по умолчанию)"
    )
    
    @validator('output_format')
    def validate_output_format(cls, v):
        if v not in ['json', 'text']:
            raise ValueError('Поддерживается только формат json или text')
        return v
    
    @validator('json_schema')
    def validate_json_schema(cls, v, values):
        if values.get('is_structured') and not v:
            raise ValueError('JSON-схема обязательна для структурированного вывода')
        return v
    
    @validator('model_id')
    def validate_model_id(cls, v):
        if v:
            # Проверяем что модель существует
            from shared.llm import get_model_by_id
            if not get_model_by_id(v):
                raise ValueError(f'Модель с ID "{v}" не найдена')
        return v


class FineTuningRequest(BaseModel):
    """Параметры для LoRA fine-tuning"""
    use_llm_judge: bool = Field(default=True, description="Использовать ли LLM Judge для оценки")
    judge_model_id: Optional[str] = Field(
        default=None,
        description="ID модели для LLM Judge (если не указан, используется модель по умолчанию)"
    )
    base_model_name: Optional[str] = Field(
        default=None,
        description="Название базовой модели для дообучения (если не указано, используется модель по умолчанию)"
    )
    n_trials: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Количество попыток байесовской оптимизации"
    )
    enable_mlflow: Optional[bool] = Field(
        default=None,
        description="Включить MLflow трекинг (если None, читается из переменных окружения)"
    )
    
    @validator('judge_model_id')
    def validate_judge_model_id(cls, v, values):
        if v and values.get('use_llm_judge'):
            # Проверяем что модель существует только если LLM Judge включен
            from shared.llm import get_model_by_id
            if not get_model_by_id(v):
                raise ValueError(f'Модель для LLM Judge с ID "{v}" не найдена')
        return v


class GenerationTaskRequest(BaseModel):
    """Запрос на создание задачи генерации"""
    project_id: int = Field(description="ID проекта")
    generation_params: GenerationRequest = Field(description="Параметры генерации")


class FineTuningTaskRequest(BaseModel):
    """Запрос на создание задачи fine-tuning"""
    project_id: int = Field(description="ID проекта")
    fine_tuning_params: FineTuningRequest = Field(description="Параметры fine-tuning")


class TaskResponse(BaseModel):
    """Ответ с информацией о задаче"""
    success: bool
    task_id: str = Field(description="ID задачи в Redis")
    message: str = Field(description="Сообщение о результате")
    queue_name: str = Field(description="Название очереди")
    model_id: Optional[str] = Field(default=None, description="ID используемой модели")


class GenerateRequest(BaseModel):
    """Запрос на генерацию данных"""
    total_results: Optional[int] = Field(
        default=100, 
        ge=1, 
        le=1000, 
        description="Количество результатов для генерации"
    )
    batch_size: Optional[int] = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="Размер батча для обработки"
    )
    max_workers: Optional[int] = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Максимальное количество потоков"
    )


class GenerateResponse(BaseModel):
    """Ответ на запрос генерации"""
    success: bool
    message: str
    output_object: Optional[str] = Field(
        None, 
        description="Путь к объекту в MinIO"
    )
    results_count: Optional[int] = Field(
        None, 
        description="Количество сгенерированных результатов"
    ) 