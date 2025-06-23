import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

# from google import genai  # TODO: Fix AI import
from .config import Config

# Импорты нашего shared API для работы с MinIO
from shared.minio import (
    get_minio_client, 
    MinIOClient,
    MinIOError, 
    ObjectNotFoundError
)

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Класс для генерации ответов через Google AI API"""

    def __init__(self, config: Config):
        self.config = config
        # self.model = genai.Client(api_key=config.api_key)  # TODO: Fix AI import
        self.model = None
        # Используем наш MinIO клиент для работы с промптами
        self.minio_client = get_minio_client()

    def _prepare_system_instruction(self, project_id: int) -> str:
        """Готовит системную инструкцию из MinIO"""
        try:
            # Загружаем базовый промпт из MinIO
            base_prompt = self.minio_client.download_text("data_generator_prompt.txt")
            
            # Загружаем системный промпт проекта из MinIO
            system_prompt_object = f"project_{project_id}/system_prompt.txt"
            system_prompt = self.minio_client.download_text(system_prompt_object)
            
            # Подставляем значения из конфига
            final_prompt = base_prompt.replace("${system_prompt_from_file}", system_prompt)
            final_prompt = final_prompt.replace("${examples_count}", str(self.config.examples_per_api_call))
            
            logger.info(f"Системный промпт для проекта {project_id} успешно подготовлен.")
            return final_prompt
            
        except (ObjectNotFoundError, MinIOError) as e:
            logger.error(f"Ошибка загрузки промптов из MinIO: {e}")
            # Fallback на локальные файлы (для совместимости)
            return self._prepare_system_instruction_fallback()

    def _prepare_system_instruction_fallback(self) -> str:
        """Fallback метод для загрузки промптов из файловой системы"""
        try:
            base_prompt = self.config.data_generator_prompt_file.read_text(encoding='utf-8')
            system_prompt = self.config.system_instruction_file.read_text(encoding='utf-8')
            
            # Подставляем значения из конфига
            final_prompt = base_prompt.replace("${system_prompt_from_file}", system_prompt)
            final_prompt = final_prompt.replace("${examples_count}", str(self.config.examples_per_api_call))
            
            logger.warning("Используется fallback загрузка промптов из файловой системы")
            return final_prompt
        except FileNotFoundError as e:
            logger.error(f"Не найден файл для системного промпта: {e}")
            raise

    def generate_new_examples(self, existing_examples: List[Dict], project_id: int = 1) -> Optional[str]:
        """
        Генерирует новые примеры данных на основе существующих.
        """
        system_instruction = self._prepare_system_instruction(project_id)
        
        try:
            # Преобразуем список словарей в одну строку JSONL для user prompt
            user_prompt = "\n".join([json.dumps(ex, ensure_ascii=False) for ex in existing_examples])
 
            logger.info(f"Отправка {len(existing_examples)} примеров для генерации новых данных...")
            
            # TODO: Fix AI import and uncomment
            # response = self.model.models.generate_content(
            #     model=self.config.model_name,
            #     contents=user_prompt,
            #     config=genai.types.GenerateContentConfig(
            #         system_instruction=system_instruction
            #     )
            # )
            
            # Временная заглушка
            response = type('MockResponse', (), {'text': '{"example": "mock data"}'})()
            logger.warning("Используется заглушка для AI генерации")
            
            logger.info("Получен ответ от API.")
            return response.text
        
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return None

    def process_batch(self, prompts: List[Dict], project_id: int = 1) -> List[Dict[str, Any]]:
        """
        Обрабатывает пакет промптов для генерации нового датасета и пытается 
        распарсить ответ в виде списка JSON-объектов.
        """
        generated_text = self.generate_new_examples(prompts, project_id)
        if not generated_text:
            logger.warning(f"Пакет из {len(prompts)} промптов обработан, но не получено результатов.")
            return []

        results = []
        clean_text = generated_text.strip()

        # Попытка очистить от Markdown, который модель может добавить
        if clean_text.startswith("```json"):
            clean_text = clean_text.lstrip("```json").strip()
        if clean_text.endswith("```"):
            clean_text = clean_text.rstrip("```").strip()

        # Сначала пытаемся распарсить весь текст как один JSON (может быть объектом или массивом)
        try:
            data = json.loads(clean_text)
            if isinstance(data, list):
                # Ответ - это JSON-массив объектов
                results.extend(data)
                logger.info(f"Успешно распарсен JSON-массив с {len(results)} объектами.")
            elif isinstance(data, dict):
                # Ответ - это один JSON-объект
                results.append(data)
                logger.info("Успешно распарсен один JSON-объект.")
            else:
                 # Неожиданный, но валидный JSON (например, просто строка или число)
                 logger.warning(f"Распарсен неожиданный тип данных: {type(data)}. Попытка обработки как JSONL.")
                 raise json.JSONDecodeError("Not a list or dict", clean_text, 0) # Переход к обработке JSONL

        except json.JSONDecodeError:
            # Если не получилось, предполагаем, что это JSONL (объекты на каждой строке)
            logger.warning("Не удалось распарсить ответ как единый JSON. Попытка обработки как JSONL...")
            for line in clean_text.split('\n'):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    results.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Не удалось распарсить строку из ответа как JSON: '{clean_line}'")
        
        if results:
            logger.info(f"Итого: успешно извлечено {len(results)} новых JSON-объектов из ответа.")
            return results
        else:
            logger.warning("Не удалось извлечь JSON из ответа. Не будет сохранено ничего.")
            return []