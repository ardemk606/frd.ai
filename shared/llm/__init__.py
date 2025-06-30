"""
Общая система для работы с LLM моделями
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class LLMProvider(str, Enum):
    """Enum для поддерживаемых провайдеров LLM"""
    GEMINI = "gemini"
    GIGACHAT = "gigachat"
    # В будущем можно добавить:
    # OPENAI = "openai"
    # ANTHROPIC = "anthropic"
    # OLLAMA = "ollama"


class BaseLLMClient(ABC):
    """Базовый абстрактный класс для всех LLM клиентов"""
    
    @abstractmethod
    def generate_content(
        self, 
        system_instruction: str, 
        contents: str, 
        model: str = None,
        max_retries: int = 3
    ) -> str:
        """
        Генерирует контент с системной инструкцией
        
        Args:
            system_instruction: Системный промпт
            contents: Пользовательский запрос
            model: Модель для использования (опционально)
            max_retries: Максимальное количество попыток
            
        Returns:
            Сгенерированный текст
            
        Raises:
            Exception: Если все попытки исчерпаны
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Возвращает модель по умолчанию для данного провайдера"""
        pass


class LLMModelInfo:
    """Информация о модели LLM"""
    
    def __init__(
        self, 
        provider: LLMProvider, 
        model_id: str, 
        display_name: str,
        description: str = "",
        is_default: bool = False
    ):
        self.provider = provider
        self.model_id = model_id
        self.display_name = display_name
        self.description = description
        self.is_default = is_default
    
    def to_dict(self) -> dict:
        """Конвертирует в словарь для API ответов"""
        return {
            "provider": self.provider.value,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "description": self.description,
            "is_default": self.is_default
        }


# Реестр доступных моделей (захардкожен как требуется)
AVAILABLE_MODELS = [
    # GigaChat модели
    LLMModelInfo(
        provider=LLMProvider.GIGACHAT,
        model_id="GigaChat",
        display_name="GigaChat",
        description="Базовая модель GigaChat от Сбера для генерации текста",
        is_default=True
    ),
    LLMModelInfo(
        provider=LLMProvider.GIGACHAT,
        model_id="GigaChat-Pro",
        display_name="GigaChat Pro",
        description="Продвинутая модель GigaChat с улучшенными возможностями"
    ),
    LLMModelInfo(
        provider=LLMProvider.GIGACHAT,
        model_id="GigaChat-Max",
        display_name="GigaChat Max",
        description="Максимальная модель GigaChat с лучшим качеством генерации"
    ),
    # Gemini модели
    LLMModelInfo(
        provider=LLMProvider.GEMINI,
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        description="Быстрая модель Google Gemini для генерации текста"
    ),
    LLMModelInfo(
        provider=LLMProvider.GEMINI,
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        description="Продвинутая модель Google Gemini с улучшенными возможностями"
    )
]


def get_available_models() -> list[LLMModelInfo]:
    """Возвращает список всех доступных моделей"""
    return AVAILABLE_MODELS


def get_default_model() -> LLMModelInfo:
    """Возвращает модель по умолчанию"""
    for model in AVAILABLE_MODELS:
        if model.is_default:
            return model
    return AVAILABLE_MODELS[0]  # Fallback на первую модель


def get_model_by_id(model_id: str) -> Optional[LLMModelInfo]:
    """Находит модель по ID"""
    for model in AVAILABLE_MODELS:
        if model.model_id == model_id:
            return model
    return None 