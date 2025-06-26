"""
Простой клиент для Google Gemini API
"""
import os
from google import genai
from google.genai import types


class GeminiClient:
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
    
    def generate_content(self, system_instruction: str, contents: str, model: str = "gemini-2.5-flash", max_retries: int = 3) -> str:
        """
        Генерирует контент с системной инструкцией
        
        Args:
            system_instruction: Системный промпт
            contents: Пользовательский запрос
            model: Модель для использования
            
        Returns:
            Сгенерированный текст
            
        Raises:
            Exception: Если все попытки исчерпаны
        """
        try:
            response = self.client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=contents
            )
        except Exception as e:
            if self.max_retries > 0:
                return self.generate_content(system_instruction, contents, model, max_retries - 1)
            else:
                raise e
                    
        return response.text
