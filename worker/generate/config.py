import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class GenerationConfig:
    """Конфигурация генерации данных"""
    # AI настройки
    api_key: str
    model_name: str = "gemini-2.5-flash-lite-preview-06-17"
    
    # Параметры генерации
    batch_size: int = 2
    total_results: int = 100
    max_workers: int = 10
    examples_per_api_call: int = 5
    
    @classmethod
    def from_env(cls) -> 'GenerationConfig':
        """Создает конфигурацию из переменных окружения"""
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY не найден в переменных окружения")
        
        return cls(
            api_key=api_key,
            model_name=os.getenv("AI_MODEL_NAME", "gemini-2.5-flash-lite-preview-06-17"),
            batch_size=int(os.getenv("BATCH_SIZE", "2")),
            total_results=int(os.getenv("TOTAL_RESULTS", "100")),
            max_workers=int(os.getenv("MAX_WORKERS", "10")),
            examples_per_api_call=int(os.getenv("EXAMPLES_PER_API_CALL", "5"))
        )


# Алиас для обратной совместимости
Config = GenerationConfig 