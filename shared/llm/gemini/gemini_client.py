"""
Простой клиент для Google Gemini API
"""
import os
import time
import random
import logging
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from shared.llm import BaseLLMClient

logger = logging.getLogger(__name__)


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
    
    def generate_content(self, system_instruction: str, contents: str, model: str = None, max_retries: int = 5) -> str:
        """
        Генерирует контент с системной инструкцией
        
        Args:
            system_instruction: Системный промпт
            contents: Пользовательский запрос
            model: Модель для использования (по умолчанию "gemini-2.5-flash")
            max_retries: Максимальное количество попыток (по умолчанию 5)
            
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
                
                # Успешное выполнение
                if attempt > 0:
                    logger.info(f"Успешное выполнение после {attempt + 1} попыток")
                
                return response.text
                
            except Exception as e:
                # Проверяем является ли это ошибкой rate limit (429)
                is_rate_limit = self._is_rate_limit_error(e)
                
                if attempt == max_retries - 1:  # Последняя попытка
                    logger.error(f"Все {max_retries} попыток исчерпаны. Последняя ошибка: {e}")
                    raise e
                
                if is_rate_limit:
                    # Для 429 ошибок используем экспоненциальный backoff с jitter
                    delay = self._calculate_backoff_delay(attempt)
                    error_details = self._get_error_details(e)
                    logger.warning(
                        f"Rate limit превышен. {error_details}. "
                        f"Попытка {attempt + 1}/{max_retries}. "
                        f"Ждем {delay:.2f}s перед повтором."
                    )
                    time.sleep(delay)
                else:
                    # Для других ошибок - короткая пауза
                    delay = 1.0 + random.uniform(0, 0.5)  # 1-1.5 секунды
                    error_details = self._get_error_details(e)
                    logger.warning(
                        f"Ошибка API: {error_details}. "
                        f"Попытка {attempt + 1}/{max_retries}. "
                        f"Ждем {delay:.2f}s перед повтором."
                    )
                    time.sleep(delay)
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Проверяет является ли ошибка связанной с rate limit (429)
        
        Args:
            error: Исключение для проверки
            
        Returns:
            True если это 429 ошибка
        """
        # Специфичная проверка для Google GenAI API ошибок
        if isinstance(error, (APIError, ClientError)):
            # Проверяем код ошибки
            if hasattr(error, 'code') and error.code == 429:
                return True
            
            # Проверяем статус
            if hasattr(error, 'status'):
                status_str = str(error.status).lower() if error.status else ""
                if 'resource_exhausted' in status_str or 'quota_exceeded' in status_str:
                    return True
        
        # Общая проверка по строке ошибки для других случаев
        error_str = str(error).lower()
        
        # Проверяем различные варианты 429 ошибок
        rate_limit_indicators = [
            '429',
            'rate limit', 
            'rate_limit',
            'quota exceeded',
            'quota_exceeded',
            'too many requests',
            'requests per second',
            'resource exhausted',
            'resource_exhausted',
            'tps',
            'rps',
            'throttled',
            'throttling'
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Вычисляет задержку для экспоненциального backoff с jitter
        
        Args:
            attempt: Номер попытки (начиная с 0)
            
        Returns:
            Время задержки в секундах
        """
        # Базовые параметры для backoff
        base_delay = 1.0  # Базовая задержка в секундах
        max_delay = 60.0  # Максимальная задержка в секундах
        jitter_factor = 0.3  # Фактор случайного смещения (30%)
        
        # Экспоненциальный рост: base_delay * (2 ^ attempt)
        exponential_delay = base_delay * (2 ** attempt)
        
        # Ограничиваем максимальной задержкой
        capped_delay = min(exponential_delay, max_delay)
        
        # Добавляем jitter для избежания thundering herd
        # Jitter: ±30% от базовой задержки
        jitter_range = capped_delay * jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        
        final_delay = max(0.1, capped_delay + jitter)  # Минимум 0.1 секунды
        
        logger.debug(f"Backoff calculation: attempt={attempt}, exponential={exponential_delay:.2f}s, "
                    f"capped={capped_delay:.2f}s, jitter={jitter:.2f}s, final={final_delay:.2f}s")
        
        return final_delay
    
    def _get_error_details(self, error: Exception) -> str:
        """
        Извлекает детальную информацию об ошибке для логирования
        
        Args:
            error: Исключение для анализа
            
        Returns:
            Строка с деталями ошибки
        """
        if isinstance(error, (APIError, ClientError)):
            details = []
            
            # Добавляем код ошибки
            if hasattr(error, 'code') and error.code:
                details.append(f"код={error.code}")
            
            # Добавляем статус
            if hasattr(error, 'status') and error.status:
                details.append(f"статус={error.status}")
            
            # Добавляем сообщение
            if hasattr(error, 'message') and error.message:
                details.append(f"сообщение='{error.message}'")
            
            if details:
                return f"Google API ошибка ({', '.join(details)})"
            else:
                return f"Google API ошибка: {str(error)}"
        else:
            # Для других типов ошибок
            return f"{type(error).__name__}: {str(error)}"
    
    def get_default_model(self) -> str:
        """Возвращает модель по умолчанию для Gemini"""
        return "gemini-2.5-flash"
