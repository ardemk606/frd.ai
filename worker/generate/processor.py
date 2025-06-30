import logging
import concurrent.futures
import math
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import Config
from .generator import ResponseGenerator

# Импорты нашего shared API
from shared.minio import (
    get_minio_client,
    DatasetService, 
    ProjectStorageService,
    MinIOError, 
    ObjectNotFoundError
)
from shared.repository import (
    DatasetRepository,
    get_database_connection,
    create_dataset_repository,
    DatasetNotFoundError, 
    RepositoryError
)

logger = logging.getLogger(__name__)


class SelfInstructProcessor:
    """Основной класс для обработки self-instruct данных"""

    def __init__(self, config: Config, system_prompt_path: str, model_id: Optional[str] = None):
        self.config = config
        self.generator = ResponseGenerator(
            system_prompt_path=system_prompt_path,
            user_prompt_path="/app/data/prompt/data_generator_prompt.txt",
            model_id=model_id  # Передаем модель в генератор
        )
        # Используем наш новый API вместо старых классов
        minio_client = get_minio_client()
        self.dataset_service = DatasetService(minio_client)
        self.project_storage_service = ProjectStorageService(minio_client)

    def process_all(self, project_id: int) -> str:
        """
        Основной метод для генерации данных. Для каждого запроса к API 
        берет случайную выборку из seed-примеров (батч).
        """
        logger.info(f"Начало процесса генерации данных для проекта {project_id}")

        try:
            # Получаем информацию о датасете из БД
            dataset_repository = create_dataset_repository()
            dataset = dataset_repository.get_by_id(project_id)
            
            # Загружаем seed примеры через наш MinIO API используя реальный object_name
            seed_examples = self.dataset_service.get_dataset_preview(dataset.object_name)
            
            if not seed_examples:
                raise ValueError(f"Не найдены seed-примеры для проекта {project_id}")
            
            logger.info(f"Загружено {len(seed_examples)} seed-примеров.")
            
        except (ObjectNotFoundError, MinIOError) as e:
            logger.error(f"Ошибка загрузки seed-примеров: {e}")
            raise ValueError(f"Не удалось загрузить данные для проекта {project_id}")
        except (DatasetNotFoundError, RepositoryError) as e:
            logger.error(f"Ошибка получения информации о датасете: {e}")
            raise ValueError(f"Проект {project_id} не найден в базе данных")

        all_results: List[Dict[str, Any]] = []

        # Вычисляем, сколько запросов нужно сделать
        required_api_calls = math.ceil(self.config.total_results / self.config.examples_per_api_call)
        logger.info(f"Требуется ~{self.config.total_results} результатов. Будет сделано {required_api_calls} запросов к API.")

        # Создаем список батчей со случайными примерами
        prompt_batches = []
        for _ in range(required_api_calls):
            if len(seed_examples) < self.config.batch_size:
                logger.warning(f"Количество seed-примеров ({len(seed_examples)}) меньше batch_size ({self.config.batch_size}). Используются все примеры.")
                sample = seed_examples
            else:
                sample = random.sample(seed_examples, self.config.batch_size)
            prompt_batches.append(sample)
        
        logger.info(f"Для {required_api_calls} запросов создано столько же батчей со случайной выборкой по {self.config.batch_size} примеров.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {executor.submit(self.generator.process_batch, batch, project_id) for batch in prompt_batches}

            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    if batch_results:
                        all_results.extend(batch_results)
                        logger.info(f"Получено {len(batch_results)} новых результатов. Всего сейчас: {len(all_results)}.")
                except Exception as e:
                    logger.error(f"Ошибка при обработке одного из запросов к API: {e}", exc_info=True)

        final_results = all_results[:self.config.total_results]
        
        # Сохраняем результаты через наш MinIO API
        try:
            output_file = self.project_storage_service.save_generation_result(
                project_id, final_results
            )
            logger.info(f"Обработка завершена. Итого сохранено: {len(final_results)} результатов.")
            
            # Обновляем статус проекта после успешной генерации
            try:
                dataset_repository = create_dataset_repository()
                dataset_repository.update_status(project_id, "READY_FOR_VALIDATION")
                logger.info(f"Статус проекта {project_id} обновлен на READY_FOR_VALIDATION")
            except Exception as e:
                logger.error(f"Ошибка обновления статуса проекта {project_id}: {e}")
                # Не прерываем выполнение, так как данные уже сохранены
            
            return output_file
        except (MinIOError, Exception) as e:
            logger.error(f"Ошибка сохранения результатов: {e}")
            raise


def main():
    """Главная функция для запуска обработки"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        config = Config.from_env()
        # Для отладки получаем системный промпт из датасета
        dataset_repository = create_dataset_repository()
        dataset = dataset_repository.get_by_id(1)
        processor = SelfInstructProcessor(config, dataset.system_prompt_object_name)
        # Для отладки используем проект ID = 1
        output_file = processor.process_all(project_id=1)
        logger.info(f"Результаты сохранены в файл: {output_file}")
    except Exception as e:
        logger.error(f"Критическая ошибка выполнения: {e}", exc_info=True)
        raise


class DataProcessor:
    """Обертка для обработки запросов генерации через Celery"""
    
    def __init__(self):
        # Используем наш repository API с подключением к БД
        pass
    
    def process_generation_request(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает запрос на генерацию данных
        
        Args:
            generation_params: Параметры генерации
        
        Returns:
            Результат генерации
        """
        try:
            logger.info(f"Обработка запроса генерации: {generation_params}")
            
            project_id = generation_params.get('project_id')
            if not project_id:
                raise ValueError("project_id не указан в параметрах генерации")
            
            # Извлекаем model_id из параметров (если указан)
            model_id = generation_params.get('model_id')
            if model_id:
                logger.info(f"Используется модель: {model_id}")
            else:
                logger.info("Используется модель по умолчанию")
            
            # Проверяем существование проекта через repository
            try:
                dataset_repository = create_dataset_repository()
                dataset = dataset_repository.get_by_id(project_id)
                logger.info(f"Найден датасет для проекта {project_id}: {dataset.object_name}")
            except DatasetNotFoundError:
                raise ValueError(f"Проект {project_id} не найден в базе данных")
            except RepositoryError as e:
                logger.error(f"Ошибка работы с базой данных: {e}")
                raise
            
            # Создаем конфигурацию на основе параметров
            config = Config.from_env()
            config.total_results = generation_params.get('examples_count', 30)
            
            # Запускаем процесс генерации с указанной моделью
            processor = SelfInstructProcessor(
                config, 
                dataset.system_prompt_object_name, 
                model_id=model_id  # Передаем модель в процессор
            )
            output_file = processor.process_all(project_id)
            
            return {
                'status': 'success',
                'output_file': output_file,
                'generated_count': config.total_results,
                'project_id': project_id,
                'model_id': model_id
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


if __name__ == "__main__":
    main()