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
    
    def generate_content(self, system_instruction: str, contents: str, model: str = "gemini-2.5-flash") -> str:
        """
        Генерирует контент с системной инструкцией
        
        Args:
            system_instruction: Системный промпт
            contents: Пользовательский запрос
            model: Модель для использования
            
        Returns:
            Сгенерированный текст
        """
        response = self.client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            ),
            contents=contents
        )
        
        return response.text
