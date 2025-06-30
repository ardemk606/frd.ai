import logging
import os
from typing import List, Dict, Any, Optional

from shared.llm.factory import get_llm_client
from shared.minio import MinIOClient
from shared.minio.dependencies import get_minio_client

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Класс для генерации ответов через LLM API"""

    def __init__(self, system_prompt_path: str, user_prompt_path: str, model_id: Optional[str] = None):
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        self.model_id = model_id
        
        # Используем фабрику для создания клиента
        self.llm_client = get_llm_client(model_id=model_id)
        self.minio_client = get_minio_client()

    def _load_prompt_from_minio(self, object_name: str) -> str:
        """Загружает промпт из MinIO"""
        try:
            content = self.minio_client.download_text(object_name)
            return content
        except Exception as e:
            logger.error(f"Error loading prompt '{object_name}': {e}")
            raise

    def generate_response(self, json_data: str, examples_count: int = 5) -> str:
        """Генерирует ответ на основе JSON данных"""
        try:
            # Получаем системный промпт из MinIO
            system_prompt = self._load_prompt_from_minio(self.system_prompt_path)
            logger.info(f"System prompt loaded from '{self.system_prompt_path}'")

            # Получаем пользовательский промпт (шаблон)
            with open(self.user_prompt_path, 'r', encoding='utf-8') as f:
                user_prompt_template = f.read()
            logger.info(f"User prompt template loaded from '{self.user_prompt_path}'")

            # Заменяем переменные в промпте
            user_prompt = user_prompt_template.replace(
                "${system_prompt_from_file}", system_prompt
            ).replace(
                "${examples_count}", str(examples_count)
            )
            
            # Финальный промпт для пользователя
            contents = f"{user_prompt}\n\n### EXISTING EXAMPLES ###\n{json_data}"
            
            logger.debug(f"Generated prompt length: {len(contents)}")
            
            # Используем LLM клиент через фабрику
            response_text = self.llm_client.generate_content(
                system_instruction=system_prompt,
                contents=contents,
                model=self.model_id  # Передаем модель явно
            )
            
            logger.debug(f"Raw response from API: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def process_batch(self, prompts: List[Dict], project_id: int = 1) -> List[Dict[str, Any]]:
        """
        Обрабатывает пакет промптов для генерации нового датасета
        """
        import json
        
        if not prompts:
            raise ValueError("Список промптов не может быть пустым")
        
        # Преобразуем список словарей в JSONL строку
        jsonl_data = "\n".join([json.dumps(prompt, ensure_ascii=False) for prompt in prompts])
        
        logger.info(f"Отправка {len(prompts)} примеров для генерации новых данных...")
        
        # Генерируем ответ
        generated_text = self.generate_response(jsonl_data, examples_count=5)
        
        if not generated_text:
            raise RuntimeError("API вернул пустой ответ")
        
        # Парсим ответ в список объектов
        results = []
        clean_text = generated_text.strip()
        
        # Очистка от Markdown
        if clean_text.startswith("```json"):
            clean_text = clean_text.lstrip("```json").strip()
        if clean_text.endswith("```"):
            clean_text = clean_text.rstrip("```").strip()
        
        # Попытка парсинга как JSON массив
        try:
            data = json.loads(clean_text)
            if isinstance(data, list):
                results.extend(data)
                logger.info(f"Успешно распарсен JSON-массив с {len(results)} объектами.")
            elif isinstance(data, dict):
                results.append(data)
                logger.info("Успешно распарсен один JSON-объект.")
            else:
                raise json.JSONDecodeError("Not a list or dict", clean_text, 0)
        except json.JSONDecodeError:
            # Парсинг как JSONL
            logger.warning("Не удалось распарсить как JSON. Попытка обработки как JSONL...")
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
            raise ValueError(f"Не удалось извлечь валидные JSON-объекты из ответа: {clean_text[:200]}...")
        
        logger.info(f"Итого: успешно извлечено {len(results)} новых JSON-объектов.")
        return results