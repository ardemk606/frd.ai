"""
Фабрика для создания LLM клиентов
"""
import logging
from typing import Optional

from shared.llm import BaseLLMClient, LLMProvider
from shared.llm.gemini.gemini_client import GeminiClient
from shared.llm.gigachat.gigachat_client import GigaChatClient

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """Фабрика для создания клиентов LLM"""
    
    @staticmethod
    def create_client(provider: LLMProvider, **kwargs) -> BaseLLMClient:
        """
        Создает клиент LLM для указанного провайдера
        
        Args:
            provider: Провайдер LLM
            **kwargs: Дополнительные параметры для клиента
            
        Returns:
            Экземпляр BaseLLMClient
            
        Raises:
            ValueError: Если провайдер не поддерживается
        """
        if provider == LLMProvider.GEMINI:
            api_key = kwargs.get('api_key')
            return GeminiClient(api_key=api_key)
        
        elif provider == LLMProvider.GIGACHAT:
            # Передаем все kwargs в GigaChatClient
            return GigaChatClient(**kwargs)
        
        # В будущем можно добавить другие провайдеры:
        # elif provider == LLMProvider.OPENAI:
        #     return OpenAIClient(**kwargs)
        # elif provider == LLMProvider.ANTHROPIC:
        #     return AnthropicClient(**kwargs)
        
        else:
            raise ValueError(f"Провайдер {provider} не поддерживается")
    
    @staticmethod
    def create_client_from_model_id(model_id: str, **kwargs) -> BaseLLMClient:
        """
        Создает клиент LLM на основе ID модели
        
        Args:
            model_id: ID модели (например, "gemini-2.5-flash")
            **kwargs: Дополнительные параметры для клиента
            
        Returns:
            Экземпляр BaseLLMClient
            
        Raises:
            ValueError: Если модель не найдена
        """
        from shared.llm import get_model_by_id
        
        model_info = get_model_by_id(model_id)
        if not model_info:
            raise ValueError(f"Модель с ID '{model_id}' не найдена")
        
        logger.info(f"Создаем клиент для модели {model_id} (провайдер: {model_info.provider})")
        return LLMClientFactory.create_client(model_info.provider, **kwargs)


def get_llm_client(model_id: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """
    Удобная функция для получения LLM клиента
    
    Args:
        model_id: ID модели (если None, используется модель по умолчанию)
        **kwargs: Дополнительные параметры для клиента
        
    Returns:
        Экземпляр BaseLLMClient
    """
    if model_id:
        return LLMClientFactory.create_client_from_model_id(model_id, **kwargs)
    else:
        # Используем модель по умолчанию
        from shared.llm import get_default_model
        default_model = get_default_model()
        return LLMClientFactory.create_client(default_model.provider, **kwargs) 