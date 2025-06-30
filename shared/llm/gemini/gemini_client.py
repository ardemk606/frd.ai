"""
Простой клиент для Google Gemini API
"""
import os
from google import genai
from google.genai import types

from shared.llm import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Простой клиент для Google Gemini API"""
    
    def __init__(self, api_key: str = None):
        """
        Инициализация клиента
        
        Args:
            api_key: API ключ Google Gemini. Если None, берется из переменной окружения
        """
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY не найден")
            
        self.client = genai.Client(api_key=api_key)
    
    def generate_content(self, system_instruction: str, contents: str, model: str = None, max_retries: int = 3) -> str:
        """
        Генерирует контент с системной инструкцией
        
        Args:
            system_instruction: Системный промпт
            contents: Пользовательский запрос
            model: Модель для использования (по умолчанию "gemini-2.5-flash")
            max_retries: Максимальное количество попыток
            
        Returns:
            Сгенерированный текст
            
        Raises:
            Exception: Если все попытки исчерпаны
        """
        if model is None:
            model = self.get_default_model()
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    contents=contents
                )
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:  # Последняя попытка
                    raise e
                # Логируем и продолжаем
                import logging
                logging.warning(f"Попытка {attempt + 1} не удалась: {e}")
    
    def get_default_model(self) -> str:
        """Возвращает модель по умолчанию для Gemini"""
        return "gemini-2.5-flash"
