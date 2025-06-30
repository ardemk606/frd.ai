"""
Клиент для GigaChat API на основе langchain-gigachat
"""
import os
import logging
from typing import Optional

from shared.llm import BaseLLMClient

logger = logging.getLogger(__name__)


class GigaChatClient(BaseLLMClient):
    """Клиент для GigaChat API (авторизация только по access_token)"""
    
    def __init__(
        self,
        access_token: Optional[str] = None,
        base_url: str = "https://gigachat.devices.sberbank.ru/api/v1",
        verify_ssl_certs: bool = False,
    ):
        """
        Инициализация клиента GigaChat
        
        Args:
            access_token: Токен доступа (альтернативная авторизация)
            base_url: Адрес API
            verify_ssl_certs: Проверка SSL сертификатов
        """
        self.base_url = base_url
        self.verify_ssl_certs = verify_ssl_certs
        self.access_token = access_token or os.environ.get("GIGACHAT_ACCESS_TOKEN")
        
        # Логируем используемый токен (частично, для безопасности)
        masked = self.access_token[:6] + "..." + self.access_token[-6:]
        logger.warning(f"[GigaChat] Используется access_token: {masked}")
        
        # Инициализируем клиент GigaChat
        self._init_client()
    
    def _init_client(self):
        """Инициализирует клиент GigaChat"""
        try:
            from langchain_gigachat.chat_models import GigaChat
            
            if not self.access_token:
                raise ValueError("GIGACHAT_ACCESS_TOKEN не указан")

            init_params = {
                "base_url": self.base_url,
                "verify_ssl_certs": self.verify_ssl_certs,
                "credentials": self.access_token,
            }
            
            self.client = GigaChat(**init_params)
            logger.info("GigaChat клиент успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации GigaChat клиента: {e}")
            raise
    
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
            model: Модель для использования (по умолчанию "GigaChat")
            max_retries: Максимальное количество попыток
            
        Returns:
            Сгенерированный текст
            
        Raises:
            Exception: Если все попытки исчерпаны
        """
        if model is None:
            model = self.get_default_model()
        
        # Формируем полный промпт с системной инструкцией
        full_prompt = f"{system_instruction}\n\n{contents}"
        
        for attempt in range(max_retries):
            try:
                # Устанавливаем модель если она указана и отличается от текущей
                if hasattr(self.client, 'model'):
                    self.client.model = model
                
                # Генерируем ответ
                logger.debug("[GigaChat] Отправка запроса invoke() ...")
                response = self.client.invoke(full_prompt)
                
                # Извлекаем текст из ответа
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
                    
            except Exception as e:
                if attempt == max_retries - 1:  # Последняя попытка
                    logger.error(f"GigaChat API ошибка после {max_retries} попыток: {e}")
                    raise e
                # Логируем и продолжаем
                logger.warning(f"GigaChat попытка {attempt + 1} не удалась: {e}")
    
    def get_default_model(self) -> str:
        """Возвращает модель по умолчанию для GigaChat"""
        return "GigaChat"
    
    def get_available_models(self) -> list:
        """Получает список доступных моделей из API"""
        try:
            if hasattr(self.client, 'get_models'):
                return self.client.get_models()
            else:
                logger.warning("Метод get_models недоступен в текущей версии langchain-gigachat")
                return ["GigaChat", "GigaChat-Pro", "GigaChat-Max"]
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            return ["GigaChat", "GigaChat-Pro", "GigaChat-Max"] 