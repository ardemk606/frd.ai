import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

from google import genai
from google.genai import types
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
    """Класс для генерации ответов через Google Gemini API"""

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.minio_client = get_minio_client()

    def _prepare_system_instruction(self, project_id: int) -> str:
        """Готовит системную инструкцию, комбинируя мета-промпт приложения и пользовательский системный промпт"""
        # Загружаем базовый мета-промпт из файловой системы контейнера
        base_prompt = self._load_meta_prompt()
        
        # Получаем информацию о проекте из БД для получения system_prompt_object_name
        from shared.repository import create_dataset_repository
        dataset_repository = create_dataset_repository()
        dataset = dataset_repository.get_by_id(project_id)
        
        if not dataset:
            raise ValueError(f"Датасет с ID {project_id} не найден")
        
        if not dataset.system_prompt_object_name:
            raise ValueError(f"У датасета {project_id} не указан system_prompt_object_name")
        
        # Загружаем пользовательский системный промпт проекта из MinIO
        system_prompt = self.minio_client.download_text(dataset.system_prompt_object_name)
        
        # Подставляем значения в мета-промпт
        final_prompt = base_prompt.replace("${system_prompt_from_file}", system_prompt)
        final_prompt = final_prompt.replace("${examples_count}", str(self.config.examples_per_api_call))
        
        logger.info(f"Системный промпт для проекта {project_id} успешно подготовлен.")
        return final_prompt

    def _load_meta_prompt(self) -> str:
        """Загружает мета-промпт из файловой системы контейнера"""
        from pathlib import Path
        
        # Единственный корректный путь в worker контейнере
        meta_prompt_path = Path("/app/data/prompt/data_generator_prompt.txt")
        
        if not meta_prompt_path.exists():
            raise FileNotFoundError(f"Мета-промпт не найден по пути: {meta_prompt_path}")
        
        logger.info(f"Загружаю мета-промпт из: {meta_prompt_path}")
        return meta_prompt_path.read_text(encoding='utf-8')

    def generate_new_examples(self, existing_examples: List[Dict], project_id: int = 1) -> str:
        """
        Генерирует новые примеры данных на основе существующих.
        """
        if not existing_examples:
            raise ValueError("Список существующих примеров не может быть пустым")
        
        system_instruction = self._prepare_system_instruction(project_id)
        
        # Преобразуем список словарей в одну строку JSONL для user prompt
        user_prompt = "\n".join([json.dumps(ex, ensure_ascii=False) for ex in existing_examples])

        logger.info(f"Отправка {len(existing_examples)} примеров для генерации новых данных...")
        
        # Вызов Google Gemini API
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self.config.temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Отключаем thinking для скорости
            )
        )
        
        if not response.text:
            raise RuntimeError("Gemini API вернул пустой ответ")
        
        logger.info("Получен ответ от Gemini API")
        return response.text

    def process_batch(self, prompts: List[Dict], project_id: int = 1) -> List[Dict[str, Any]]:
        """
        Обрабатывает пакет промптов для генерации нового датасета и пытается 
        распарсить ответ в виде списка JSON-объектов.
        """
        generated_text = self.generate_new_examples(prompts, project_id)
        
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
            for line_num, line in enumerate(clean_text.split('\n'), 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    results.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Строка {line_num} не является валидным JSON: '{clean_line}' - {e}")
        
        if not results:
            raise ValueError(f"Не удалось извлечь ни одного валидного JSON-объекта из ответа Gemini: {clean_text[:200]}...")
        
        logger.info(f"Итого: успешно извлечено {len(results)} новых JSON-объектов из ответа.")
        return results