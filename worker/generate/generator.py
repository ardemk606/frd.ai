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
        """Готовит системную инструкцию, комбинируя мета-промпт приложения и пользовательский системный промпт"""
        try:
            # Загружаем базовый мета-промпт из файловой системы контейнера (ресурс приложения)
            base_prompt = self._load_meta_prompt()
            
            # Получаем информацию о проекте из БД для получения system_prompt_object_name
            from shared.repository import create_dataset_repository
            dataset_repository = create_dataset_repository()
            dataset = dataset_repository.get_by_id(project_id)
            
            if dataset.system_prompt_object_name:
                # Загружаем пользовательский системный промпт проекта из MinIO
                system_prompt = self.minio_client.download_text(dataset.system_prompt_object_name)
            else:
                # Fallback если нет системного промпта
                system_prompt = "You are an AI assistant that generates training data."
            
            # Подставляем значения в мета-промпт
            final_prompt = base_prompt.replace("${system_prompt_from_file}", system_prompt)
            final_prompt = final_prompt.replace("${examples_count}", str(self.config.examples_per_api_call))
            
            logger.info(f"Системный промпт для проекта {project_id} успешно подготовлен.")
            return final_prompt
            
        except Exception as e:
            logger.error(f"Ошибка подготовки системного промпта: {e}")
            return self._prepare_system_instruction_fallback()

    def _load_meta_prompt(self) -> str:
        """Загружает мета-промпт из файловой системы контейнера"""
        from pathlib import Path
        
        # Пути где может находиться мета-промпт в контейнере
        possible_paths = [
            Path("/app/data/prompt/data_generator_prompt.txt"),  # Основной путь в worker контейнере
            Path("data/prompt/data_generator_prompt.txt"),       # Относительный путь
            Path("../data/prompt/data_generator_prompt.txt"),    # На уровень выше
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Загружаю мета-промпт из: {path}")
                return path.read_text(encoding='utf-8')
        
        # Если файл не найден, возвращаем базовый встроенный промпт
        logger.warning("Мета-промпт не найден в файловой системе, используется встроенный")
        return self._get_builtin_meta_prompt()

    def _get_builtin_meta_prompt(self) -> str:
        """Возвращает встроенный мета-промпт как fallback"""
        return """### ROLE & GOAL ###
You are an expert AI assistant specialized in generating high-quality training data. Your goal is to create new, unique training examples.

### PRIMARY TASK ###
Generate ${examples_count} new JSON objects containing "input" and "output" pairs that follow the persona defined below.

### CRITICAL RULES ###
1. **ADHERE TO THE PERSONA:** The "output" must perfectly represent the persona described.
2. **PRESERVE MEANING:** The "output" must be a stylistic translation of the "input", preserving the original meaning.
3. **BE ORIGINAL:** Create completely new scenarios, substantially different from the examples provided.
4. **OUTPUT FORMAT:** Your entire response MUST be valid JSON. Return an array of objects or one object per line.

---

### SYSTEM PROMPT FOR THE PERSONA ###
${system_prompt_from_file}

---

### YOUR GENERATION TASK ###
Generate ${examples_count} new JSON objects based on the examples provided in the user input."""

    def _prepare_system_instruction_fallback(self) -> str:
        """Fallback метод когда все остальное не работает"""
        logger.warning("Используется аварийный fallback промпт")
        return f"""You are an AI assistant that generates {self.config.examples_per_api_call} new training examples based on the provided examples. 
Generate JSON objects in the same format as the input examples."""

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
            
            # Временно используем реальные данные из mock файла
            mock_data = self._load_mock_data()
            logger.info("Используются данные из mock/output.jsonl для имитации AI генерации")
            return mock_data
        
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return None

    def _load_mock_data(self) -> str:
        """Загружает мок данные из файла для имитации AI генерации"""
        try:
            from pathlib import Path
            import random
            
            # Ищем mock файл
            possible_paths = [
                Path("/app/mock/output.jsonl"),
                Path("mock/output.jsonl"),
                Path("../mock/output.jsonl"),
            ]
            
            mock_file = None
            for path in possible_paths:
                if path.exists():
                    mock_file = path
                    break
            
            if not mock_file:
                logger.warning("Mock файл не найден, используется простой мок")
                return '{"input": "Пример входа", "output": "Пример выхода"}'
            
            # Читаем все строки
            lines = mock_file.read_text(encoding='utf-8').strip().split('\n')
            
            # Берем случайные строки в количестве examples_per_api_call
            sample_size = min(self.config.examples_per_api_call, len(lines))
            selected_lines = random.sample(lines, sample_size)
            
            # Возвращаем как JSONL
            result = '\n'.join(selected_lines)
            logger.info(f"Загружено {sample_size} примеров из mock файла")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка загрузки mock данных: {e}")
            # Fallback мок
            mock_examples = []
            for i in range(self.config.examples_per_api_call):
                mock_examples.append(f'{{"input": "Пример входа {i+1}", "output": "Пример выхода {i+1}"}}')
            return '\n'.join(mock_examples)

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