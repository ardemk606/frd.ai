"""
Роутер для управления моделями LLM
"""
from fastapi import APIRouter, HTTPException
from typing import List

from shared.llm import get_available_models, get_default_model, get_model_by_id

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/available", summary="Получить список доступных моделей")
async def get_available_models_endpoint():
    """
    Возвращает список всех доступных моделей LLM
    
    Returns:
        Список моделей с их характеристиками
    """
    try:
        models = get_available_models()
        return {
            "success": True,
            "models": [model.to_dict() for model in models],
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения моделей: {str(e)}")


@router.get("/default", summary="Получить модель по умолчанию")
async def get_default_model_endpoint():
    """
    Возвращает модель по умолчанию
    
    Returns:
        Информация о модели по умолчанию
    """
    try:
        default_model = get_default_model()
        return {
            "success": True,
            "model": default_model.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения модели по умолчанию: {str(e)}")


@router.get("/{model_id}", summary="Получить информацию о конкретной модели")
async def get_model_by_id_endpoint(model_id: str):
    """
    Возвращает информацию о модели по её ID
    
    Args:
        model_id: Идентификатор модели
        
    Returns:
        Информация о модели
    """
    try:
        model = get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена")
        
        return {
            "success": True,
            "model": model.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения модели: {str(e)}") 