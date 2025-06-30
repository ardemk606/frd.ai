"""
Валидатор датасетов с метриками качества
"""
import json
import logging
from typing import Dict, List, Any, Tuple
from collections import Counter
import re

# Для BLEU метрики
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from celery import current_task

# Импорты нашего shared API
from shared.minio import (
    get_minio_client,
    DatasetService,
    MinIOError,
    ObjectNotFoundError
)
from shared.repository import (
    create_dataset_repository,
    DatasetNotFoundError,
    RepositoryError
)
from shared.minio.models import JSONLUploadRequest, TextUploadRequest

logger = logging.getLogger(__name__)

# Загружаем NLTK данные если нужно
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class DatasetValidator:
    """Валидатор для проверки качества сгенерированных датасетов"""
    
    def __init__(self):
        """Инициализация валидатора"""
        self.minio_client = get_minio_client()
        self.dataset_service = DatasetService(self.minio_client)
        
        # Пороги валидации
        self.max_bleu_threshold = 0.4        # Максимальный BLEU между input/output
        self.min_uniqueness_threshold = 0.8  # Минимальная уникальность
        self.min_valid_json_ratio = 0.95     # Минимальный процент валидных JSON
    
    def validate_dataset(self, validation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод валидации датасета
        
        Args:
            validation_params: Параметры валидации
            
        Returns:
            Результат валидации с метриками
        """
        dataset_id = validation_params['dataset_id']
        output_type = validation_params.get('output_type', 'TEXT')
        schema = validation_params.get('schema')
        
        try:
            # Получаем информацию о датасете
            dataset_repository = create_dataset_repository()
            dataset = dataset_repository.get_by_id(dataset_id)
            
            current_task.update_state(
                state='PROGRESS',
                meta={'current': 10, 'total': 100, 'status': 'Загрузка данных из MinIO...'}
            )
            
            # Загружаем сгенерированные данные
            generated_data = self._load_generated_data(dataset_id)
            
            current_task.update_state(
                state='PROGRESS',
                meta={'current': 30, 'total': 100, 'status': f'Анализ {len(generated_data)} записей...'}
            )
            
            # Запускаем валидацию
            validation_result = self._validate_data(
                generated_data, 
                output_type, 
                schema
            )
            
            current_task.update_state(
                state='PROGRESS',
                meta={'current': 90, 'total': 100, 'status': 'Обновление статуса...'}
            )
            
            # Определяем финальный статус
            is_valid = self._is_dataset_valid(validation_result)
            final_status = "READY_FOR_FINE_TUNING" if is_valid else "VALIDATION_FAILED"
            
            # Сохраняем очищенную версию датасета если валидация прошла
            if is_valid and validation_result['valid_records']:
                current_task.update_state(
                    state='PROGRESS',
                    meta={'current': 95, 'total': 100, 'status': 'Сохранение очищенного датасета...'}
                )
                
                cleaned_file_path = self._save_cleaned_dataset(
                    dataset_id, 
                    validation_result['valid_records']
                )
                logger.info(f"Очищенный датасет сохранен: {cleaned_file_path}")
                
                # Также сохраняем отчет о фильтрации
                if validation_result['filtered_records']:
                    filtered_report_path = self._save_filtering_report(
                        dataset_id,
                        validation_result['filtered_records'],
                        validation_result
                    )
                    logger.info(f"Отчет о фильтрации сохранен: {filtered_report_path}")
            
            # Обновляем статус в БД
            dataset_repository.update_status(dataset_id, final_status)
            
            logger.info(f"Валидация датасета {dataset_id} завершена. Статус: {final_status}")
            
            current_task.update_state(
                state='PROGRESS',
                meta={'current': 100, 'total': 100, 'status': 'Готово!'}
            )
            
            return {
                'dataset_id': dataset_id,
                'final_status': final_status,
                'is_valid': is_valid,
                'metrics': validation_result,
                'total_records': len(generated_data)
            }
            
        except Exception as e:
            logger.error(f"Ошибка валидации датасета {dataset_id}: {e}")
            # Устанавливаем статус ошибки
            try:
                dataset_repository = create_dataset_repository()
                dataset_repository.update_status(dataset_id, "VALIDATION_FAILED")
            except:
                pass
            raise
    
    def _load_generated_data(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Загружает сгенерированные данные из MinIO"""
        try:
            # Ищем файл сгенерированных данных
            prefix = f"project_{dataset_id}/generated/"
            
            # Получаем список объектов с этим префиксом
            objects = self.minio_client.list_objects(prefix)
            
            if not objects:
                raise ValueError(f"Не найдены сгенерированные файлы с префиксом: {prefix}")
            
            # Ищем файл с паттерном output_*.jsonl
            target_object = None
            for obj in objects:
                if obj.object_name.endswith('.jsonl') and 'output_' in obj.object_name:
                    target_object = obj
                    break
            
            if not target_object:
                raise ValueError(f"Не найден output файл в папке: {prefix}")
            
            # Загружаем полный датасет
            data = self.dataset_service.get_full_dataset(target_object.object_name)
            
            if not data:
                raise ValueError(f"Сгенерированный датасет пуст: {target_object.object_name}")
            
            logger.info(f"Загружено {len(data)} записей из {target_object.object_name}")
            return data
                
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки сгенерированных данных для датасета {dataset_id}: {e}")
            raise RuntimeError(f"Не удалось загрузить сгенерированные данные: {str(e)}") from e
    
    def _validate_data(self, data: List[Dict[str, Any]], output_type: str, schema: str = None) -> Dict[str, Any]:
        """Проводит валидацию данных с различными метриками"""
        
        total_records = len(data)
        valid_records = []  # Сюда будем складывать валидные записи
        filtered_records = []  # Записи, которые были отфильтрованы
        
        valid_json_count = 0
        bleu_scores = []
        input_texts = []
        output_texts = []
        
        # Анализируем каждую запись
        for i, record in enumerate(data):
            if i % 10 == 0:  # Обновляем прогресс каждые 10 записей
                current_task.update_state(
                    state='PROGRESS',
                    meta={
                        'current': 30 + int((i / total_records) * 50),
                        'total': 100,
                        'status': f'Анализ записи {i+1}/{total_records}...'
                    }
                )
            
            # Флаг валидности записи
            is_record_valid = True
            rejection_reasons = []
            
            # 1. Проверка структуры JSON
            if not self._is_valid_json_record(record):
                is_record_valid = False
                rejection_reasons.append("invalid_json_structure")
            else:
                valid_json_count += 1
                
                input_text = record.get('input', '')
                output_text = record.get('output', '')
                
                if input_text and output_text:
                    # 2. Вычисляем BLEU между input и output
                    bleu_score = self._calculate_bleu(input_text, output_text)
                    
                    # Проверяем BLEU порог для отдельной записи
                    if bleu_score > self.max_bleu_threshold:
                        is_record_valid = False
                        rejection_reasons.append(f"high_bleu_{bleu_score:.3f}")
                    
                    # 3. JSON валидация только если нужна
                    if output_type == 'JSON' and schema:
                        if not self._validate_json_output(output_text, schema):
                            is_record_valid = False
                            rejection_reasons.append("invalid_json_output")
                    
                    # 4. Проверка на дубликаты input'ов (базовая)
                    if input_text in input_texts:
                        is_record_valid = False
                        rejection_reasons.append("duplicate_input")
                    
                    # Если запись валидна, добавляем в финальные данные
                    if is_record_valid:
                        input_texts.append(input_text)
                        output_texts.append(output_text)
                        bleu_scores.append(bleu_score)
                        valid_records.append(record)
                    else:
                        # Добавляем информацию о причине отклонения
                        rejected_record = record.copy()
                        rejected_record['_rejection_reasons'] = rejection_reasons
                        rejected_record['_bleu_score'] = bleu_score
                        filtered_records.append(rejected_record)
                else:
                    is_record_valid = False
                    rejection_reasons.append("empty_input_or_output")
        
        # Рассчитываем метрики для финальных данных
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        valid_json_ratio = valid_json_count / total_records if total_records > 0 else 0
        
        # Уникальность input'ов (теперь гарантированно 100% для валидных записей)
        input_uniqueness = len(set(input_texts)) / len(input_texts) if input_texts else 0
        
        # Уникальность пар input-output
        pair_uniqueness = len(set(zip(input_texts, output_texts))) / len(input_texts) if input_texts else 0
        
        # Количество записей с высоким BLEU (должно быть 0 после фильтрации)
        high_bleu_count = sum(1 for score in bleu_scores if score > self.max_bleu_threshold)
        
        return {
            'total_records': total_records,
            'valid_records_count': len(valid_records),
            'filtered_records_count': len(filtered_records),
            'valid_records': valid_records,  # Чистые данные для fine-tuning
            'filtered_records': filtered_records,  # Отфильтрованные записи для анализа
            'valid_json_ratio': valid_json_ratio,
            'avg_bleu_score': avg_bleu,
            'high_bleu_count': high_bleu_count,
            'input_uniqueness': input_uniqueness,
            'pair_uniqueness': pair_uniqueness,
            'bleu_scores': bleu_scores,
            'thresholds': {
                'max_bleu': self.max_bleu_threshold,
                'min_uniqueness': self.min_uniqueness_threshold,
                'min_valid_json': self.min_valid_json_ratio
            }
        }
    
    def _is_valid_json_record(self, record: Dict[str, Any]) -> bool:
        """Проверяет что запись содержит необходимые поля"""
        return (
            isinstance(record, dict) and
            'input' in record and
            'output' in record and
            isinstance(record['input'], str) and
            isinstance(record['output'], str) and
            len(record['input'].strip()) > 0 and
            len(record['output'].strip()) > 0
        )
    
    def _calculate_bleu(self, input_text: str, output_text: str) -> float:
        """Вычисляет BLEU score между input и output"""
        try:
            # Токенизируем тексты
            input_tokens = input_text.lower().split()
            output_tokens = output_text.lower().split()
            
            if not input_tokens or not output_tokens:
                return 0.0
            
            # Используем сглаживание для коротких текстов
            smoothing = SmoothingFunction().method1
            
            # Вычисляем BLEU с 1-4 граммами
            bleu = sentence_bleu(
                [input_tokens], 
                output_tokens,
                smoothing_function=smoothing
            )
            
            return bleu
            
        except Exception as e:
            logger.warning(f"Ошибка вычисления BLEU: {e}")
            return 0.0
    
    def _validate_json_output(self, output_text: str, schema: str = None) -> bool:
        """Валидирует JSON в output тексте"""
        try:
            # Пытаемся распарсить как JSON
            json_data = json.loads(output_text)
            
            # TODO: Добавить валидацию по JSON схеме если нужно
            if schema:
                # Здесь можно добавить jsonschema валидацию
                pass
            
            return True
            
        except json.JSONDecodeError:
            return False
    
    def _is_dataset_valid(self, validation_result: Dict[str, Any]) -> bool:
        """Определяет прошел ли датасет валидацию"""
        
        total_records = validation_result['total_records']
        valid_records_count = validation_result['valid_records_count']
        
        # Минимальное количество валидных записей (абсолютное)
        min_valid_records = 10  # Хотя бы 10 записей должно остаться
        
        # Минимальный процент сохраненных записей (относительное)
        min_retention_ratio = 0.3  # Хотя бы 30% записей должно остаться
        
        # Проверяем критерии
        enough_records = valid_records_count >= min_valid_records
        good_retention = (valid_records_count / total_records) >= min_retention_ratio if total_records > 0 else False
        
        # Проверяем метрики качества для очищенных данных
        valid_json_ok = validation_result['valid_json_ratio'] >= self.min_valid_json_ratio
        bleu_ok = validation_result['avg_bleu_score'] <= self.max_bleu_threshold
        uniqueness_ok = validation_result['input_uniqueness'] >= self.min_uniqueness_threshold
        
        logger.info(f"Результаты валидации:")
        logger.info(f"  Исходных записей: {total_records}")
        logger.info(f"  Валидных записей: {valid_records_count}")
        logger.info(f"  Отфильтровано: {validation_result['filtered_records_count']}")
        logger.info(f"  Процент сохранения: {(valid_records_count / total_records) * 100:.1f}%")
        logger.info(f"  Достаточно записей: {valid_records_count} >= {min_valid_records} : {enough_records}")
        logger.info(f"  Хороший retention: {(valid_records_count / total_records):.3f} >= {min_retention_ratio} : {good_retention}")
        logger.info(f"  Valid JSON: {validation_result['valid_json_ratio']:.3f} >= {self.min_valid_json_ratio} : {valid_json_ok}")
        logger.info(f"  Avg BLEU: {validation_result['avg_bleu_score']:.3f} <= {self.max_bleu_threshold} : {bleu_ok}")
        logger.info(f"  Uniqueness: {validation_result['input_uniqueness']:.3f} >= {self.min_uniqueness_threshold} : {uniqueness_ok}")
        
        return enough_records and good_retention and valid_json_ok and bleu_ok and uniqueness_ok
    
    def _save_cleaned_dataset(self, dataset_id: int, valid_records: List[Dict[str, Any]]) -> str:
        """Сохраняет очищенную версию датасета в MinIO"""
        try:
            import datetime
            
            # Генерируем имя файла для очищенного датасета
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            object_name = f"project_{dataset_id}/validated/cleaned_dataset_{timestamp}.jsonl"
            
            # Убираем служебные поля если есть
            clean_records = []
            for record in valid_records:
                clean_record = {k: v for k, v in record.items() if not k.startswith('_')}
                clean_records.append(clean_record)
            
            # Нормализуем system промпты перед сохранением
            clean_records = self._normalize_system_prompts(clean_records)
            
            # Создаем запрос для upload_jsonl
            upload_request = JSONLUploadRequest(
                object_name=object_name,
                json_objects=clean_records,
                metadata={
                    "dataset_id": str(dataset_id),
                    "type": "cleaned_dataset",
                    "timestamp": timestamp,
                    "record_count": str(len(clean_records))
                }
            )
            
            # Сохраняем через наш MinIO клиент
            result = self.minio_client.upload_jsonl(upload_request)
            
            logger.info(f"Сохранено {len(clean_records)} очищенных записей в {object_name}")
            return object_name
            
        except Exception as e:
            logger.error(f"Ошибка сохранения очищенного датасета: {e}")
            raise
    
    def _save_filtering_report(self, dataset_id: int, filtered_records: List[Dict[str, Any]], validation_result: Dict[str, Any]) -> str:
        """Сохраняет отчет о фильтрации записей"""
        try:
            import datetime
            from collections import Counter
            import json
            
            # Генерируем имя файла для отчета
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            object_name = f"project_{dataset_id}/validation_reports/filtering_report_{timestamp}.json"
            
            # Анализируем причины отклонения
            rejection_reasons = []
            for record in filtered_records:
                if '_rejection_reasons' in record:
                    rejection_reasons.extend(record['_rejection_reasons'])
            
            reason_counts = Counter(rejection_reasons)
            
            # Создаем детальный отчет
            report = {
                'dataset_id': dataset_id,
                'validation_timestamp': timestamp,
                'summary': {
                    'total_original_records': validation_result['total_records'],
                    'valid_records': validation_result['valid_records_count'],
                    'filtered_records': validation_result['filtered_records_count'],
                    'filtering_ratio': validation_result['filtered_records_count'] / validation_result['total_records']
                },
                'metrics': {
                    'avg_bleu_score': validation_result['avg_bleu_score'],
                    'input_uniqueness': validation_result['input_uniqueness'],
                    'pair_uniqueness': validation_result['pair_uniqueness'],
                    'valid_json_ratio': validation_result['valid_json_ratio']
                },
                'thresholds_used': validation_result['thresholds'],
                'rejection_reasons_summary': dict(reason_counts),
                'filtered_records_sample': filtered_records[:10],  # Первые 10 для примера
                'bleu_score_distribution': {
                    'min': min(validation_result['bleu_scores']) if validation_result['bleu_scores'] else 0,
                    'max': max(validation_result['bleu_scores']) if validation_result['bleu_scores'] else 0,
                    'avg': validation_result['avg_bleu_score']
                }
            }
            
            # Создаем запрос для upload_text
            report_json = json.dumps(report, ensure_ascii=False, indent=2)
            upload_request = TextUploadRequest(
                object_name=object_name,
                text_content=report_json,
                metadata={
                    "dataset_id": str(dataset_id),
                    "type": "filtering_report",
                    "timestamp": timestamp,
                    "filtered_count": str(len(filtered_records))
                }
            )
            
            # Сохраняем через наш MinIO клиент
            result = self.minio_client.upload_text(upload_request)
            
            logger.info(f"Отчет о фильтрации сохранен в MinIO: {object_name}")
            return object_name
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении отчета о фильтрации: {e}")
            raise RuntimeError(f"Не удалось сохранить отчет о фильтрации: {str(e)}") from e

    def _normalize_system_prompts(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Нормализует system промпты в записях:
        1. Добавляет system: "$systemPrompt" для записей без system
        2. Заменяет реальные system промпты на плейсхолдер "$systemPrompt"
        
        Args:
            records: Список записей датасета
            
        Returns:
            Список записей с нормализованными system промптами
        """
        normalized_records = []
        
        for record in records:
            normalized_record = record.copy()
            
            # Проверяем наличие поля system
            if 'system' not in normalized_record:
                # Случай 1: Нет system поля - добавляем плейсхолдер
                normalized_record['system'] = "$systemPrompt"
                logger.debug("Добавлен плейсхолдер system для записи без system поля")
            elif normalized_record['system'] != "$systemPrompt":
                # Случай 2: Есть реальный system промпт - заменяем на плейсхолдер
                logger.debug(f"Заменен реальный system промпт на плейсхолдер")
                normalized_record['system'] = "$systemPrompt"
            # Случай 3: Уже есть плейсхолдер - ничего не делаем
            
            normalized_records.append(normalized_record)
        
        logger.info(f"Нормализованы system промпты для {len(records)} записей")
        return normalized_records 