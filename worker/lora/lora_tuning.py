import os
# Отключаем JIT компиляцию CUDA чтобы избежать проблем с Python.h
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, List
import json
import tempfile
import logging

# Импорты из того же пакета
try:
    from .bayesian_optimizer import BayesianOptimizer
    from .lora_tuning_config import LoRATuningConfig
    from .evaluation import ModelEvaluator
except ImportError:
    # Если запускаем напрямую, импортируем абсолютно
    from lora.bayesian_optimizer import BayesianOptimizer
    from lora.lora_tuning_config import LoRATuningConfig
    from lora.evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


def load_model(model_name: str = "Qwen/Qwen3-0.6B"):
    """Загружает модель
    
    Args:
        model_name: Название модели для загрузки
    """
    
    logger.info(f"Загрузка модели: {model_name}")
    
    try:
        # Загружаем токенизатор
        logger.info("Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Установлен pad_token = eos_token")
        
        logger.info("Токенизатор загружен")
        
        # Загружаем модель
        logger.info("Загрузка модели...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        logger.info("Модель загружена")
        
        # Определяем устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Устройство: {device}")
        
        model = model.to(device)
        logger.info(f"Модель перенесена на {device}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}", exc_info=True)
        raise

def setup_lora_model(model, lora_params: Dict[str, Any]):
    """Настраивает LoRA для модели"""
    # Проверяем, есть ли уже PEFT адаптеры в модели
    if hasattr(model, 'peft_config') and model.peft_config:
        logger.info("Найдены существующие PEFT адаптеры, выгружаем их...")
        try:
            model = model.unload()
            logger.info("PEFT адаптеры успешно выгружены")
        except Exception as e:
            logger.warning(f"Не удалось выгрузить PEFT адаптеры: {e}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_params['rank'],
        lora_alpha=lora_params['lora_alpha'],
        lora_dropout=lora_params['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    peft_model = get_peft_model(model, lora_config)
    return peft_model

def prepare_dataset_with_split(data: List[Dict[str, str]], tokenizer, system_prompt: str = None, val_ratio: float = 0.2):
    """Подготавливает датасет с разделением на train/val"""
    import random
    from datasets import Dataset
    
    # Перемешиваем данные
    random.shuffle(data)
    
    # Разделяем на train/val
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    logger.info(f"Разделение датасета: {len(train_data)} train + {len(val_data)} validation")
    
    # Подготавливаем train dataset
    train_dataset = prepare_dataset(train_data, tokenizer, system_prompt)
    
    return train_dataset, val_data


def prepare_dataset(data: List[Dict[str, str]], tokenizer, system_prompt: str = None):
    """Подготавливает датасет для обучения"""
    from datasets import Dataset
    
    # Подготавливаем тексты в формате для causal LM
    texts = []
    for item in data:
        # Заменяем плейсхолдер системного промпта на реальный, если он есть
        system_text = ""
        if 'system' in item:
            if item['system'] == "$systemPrompt" and system_prompt:
                # Заменяем плейсхолдер на реальный системный промпт
                system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            elif item['system'] != "$systemPrompt":
                # Используем системный промпт из записи (если он не плейсхолдер)
                system_text = f"<|im_start|>system\n{item['system']}<|im_end|>\n"
        
        # Формируем полный текст
        text = f"{system_text}<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        texts.append(text)
    
    # Создаем Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Токенизируем
    def tokenize_function(examples):
        # Токенизируем с правильными параметрами
        tokens = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        # Добавляем labels для causal language modeling (копируем input_ids)
        tokens["labels"] = tokens["input_ids"]
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def fine_tune_lora(model, tokenizer, dataset, lora_params: Dict[str, Any], output_dir: str, config: LoRATuningConfig = None):
    """Дообучает модель с LoRA
    
    Args:
        model: Базовая модель
        tokenizer: Токенизатор
        dataset: Подготовленный датасет
        lora_params: Параметры LoRA (rank, alpha, dropout, learning_rate)
        output_dir: Директория для сохранения
        config: Конфигурация обучения
    """
    logger.info(f"Начинаем дообучение LoRA в {output_dir}")
    
    # Создаем PEFT модель
    peft_model = setup_lora_model(model, lora_params)
    
    # Настройки обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=lora_params['learning_rate'],
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        warmup_steps=config.warmup_steps,
        remove_unused_columns=False,
        dataloader_drop_last=True,  # Избегаем проблем с разными размерами батчей
        fp16=True,  # Используем mixed precision для экономии памяти
        label_names=["labels"],  # Явно указываем колонку с ответами
    )
    
    # Создаем тренер с совместимым API
    import inspect
    trainer_signature = inspect.signature(Trainer.__init__)
    
    if 'processing_class' in trainer_signature.parameters:
        # Новая версия transformers - используем processing_class
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        # Старая версия transformers - используем tokenizer 
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*tokenizer.*deprecated.*")
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
            )
    
    # Запускаем обучение
    trainer.train()
    
    # Сохраняем адаптер
    peft_model.save_pretrained(output_dir)
    
    return peft_model


def safe_unload_peft_adapters(model):
    """Безопасно выгружает PEFT адаптеры из модели"""
    try:
        # Проверяем наличие PEFT конфигурации
        if not hasattr(model, 'peft_config') or not model.peft_config:
            logger.debug("Модель не содержит PEFT адаптеры")
            return model
            
        logger.info("Найдены PEFT адаптеры, пытаемся их выгрузить...")
        
        # Пробуем различные методы выгрузки адаптеров
        if hasattr(model, 'unload_and_optionally_merge'):
            logger.info("Выгружаем PEFT адаптеры методом unload_and_optionally_merge()")
            base_model = model.unload_and_optionally_merge(merge=False)
            return base_model
            
        elif hasattr(model, 'merge_and_unload'):
            logger.info("Выгружаем PEFT адаптеры методом merge_and_unload()")
            base_model = model.merge_and_unload()
            return base_model
            
        elif hasattr(model, 'unload'):
            logger.info("Выгружаем PEFT адаптеры методом unload()")
            model.unload()
            return model
            
        elif hasattr(model, 'disable_adapters'):
            logger.info("Отключаем PEFT адаптеры методом disable_adapters()")
            model.disable_adapters()
            return model
            
        else:
            # Последний вариант - получаем базовую модель напрямую
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                logger.info("Получаем базовую модель напрямую через base_model.model")
                return model.base_model.model
            elif hasattr(model, 'model'):
                logger.info("Получаем базовую модель напрямую через model")
                return model.model
            else:
                logger.warning("Не найден способ выгрузки PEFT адаптеров, продолжаем с текущей моделью")
                return model
                
    except Exception as e:
        logger.warning(f"Не удалось выгрузить PEFT адаптеры: {e}")
        return model


def cleanup_peft_model(peft_model):
    """Полная очистка PEFT модели и освобождение памяти"""
    try:
        # Выгружаем адаптеры (но не присваиваем результат, так как модель будет удалена)
        try:
            safe_unload_peft_adapters(peft_model)
        except Exception as unload_error:
            logger.warning(f"Ошибка при выгрузке адаптеров: {unload_error}")
        
        # Перемещаем модель на CPU для освобождения GPU памяти
        try:
            if hasattr(peft_model, 'cpu'):
                peft_model.cpu()
        except Exception as cpu_error:
            logger.warning(f"Ошибка при перемещении модели на CPU: {cpu_error}")
        
        # Удаляем ссылки
        del peft_model
        
        # Принудительная очистка памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        logger.debug("PEFT модель очищена из памяти")
        
    except Exception as e:
        logger.warning(f"Ошибка при очистке PEFT модели: {e}")


class LoRATuner:
    """Основной класс для дообучения LoRA с байесовской оптимизацией"""
    
    def __init__(self, config: LoRATuningConfig = None, judge_model_id: str = None, use_llm_judge: bool = True, enable_mlflow: bool = None):
        """
        Args:
            config: Конфигурация для дообучения
            judge_model_id: ID модели для LLM Judge оценки
            use_llm_judge: Использовать ли LLM Judge для оценки (по умолчанию True)
            enable_mlflow: Включить ли MLflow трекинг (если None, читается из env)
        """
        self.config = config or LoRATuningConfig.from_env()
        self.use_llm_judge = use_llm_judge
        self.optimizer = BayesianOptimizer(n_trials=self.config.n_trials)
        
        # Создаем evaluator только если нужен LLM Judge
        if self.use_llm_judge:
            self.evaluator = ModelEvaluator(judge_model_id, use_llm_judge=True)
            logger.info(f"LLM Judge включен с моделью: {judge_model_id or 'по умолчанию'}")
        else:
            self.evaluator = ModelEvaluator(judge_model_id=None, use_llm_judge=False)
            logger.info("LLM Judge выключен, будет использоваться только BERTScore")
        
        # Инициализация MLflow трекера
        try:
            from .mlflow_tracker import create_mlflow_tracker
            self.mlflow_tracker = create_mlflow_tracker(enabled=enable_mlflow)
            if self.mlflow_tracker.enabled:
                logger.info("MLflow трекинг включен для эксперимента")
            else:
                logger.info("MLflow трекинг отключен")
        except ImportError:
            logger.info("MLflow трекер недоступен")
            self.mlflow_tracker = None
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.validation_data = None
        self.system_prompt = None
        
        # Для отслеживания экспериментов
        self.experiment_metadata = {
            "dataset_size": 0,
            "validation_size": 0,
            "model_name": None,
            "use_llm_judge": use_llm_judge,
            "judge_model_id": judge_model_id
        }

    def load_data(self, data_path: str, model_name: str = None, system_prompt: str = None) -> None:
        """Загружает данные для обучения
        
        Args:
            data_path: Путь к файлу с данными
            model_name: Название модели для загрузки
            system_prompt: Системный промпт для замены плейсхолдеров
        """
        logger.info(f"Загрузка данных из {data_path}...")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            logger.info(f"Загружено {len(data)} примеров из {data_path}")
            
            # Проверяем минимальное количество примеров
            if len(data) < 5:
                logger.warning(f"⚠️  МАЛО ДАННЫХ: Загружено только {len(data)} примеров! "
                              f"Рекомендуется минимум 50-100 примеров для качественного дообучения.")
            elif len(data) < 20:
                logger.warning(f"⚠️  НЕБОЛЬШОЙ ДАТАСЕТ: Загружено {len(data)} примеров. "
                              f"Для лучших результатов рекомендуется 50-100+ примеров.")
            else:
                logger.info(f"✅ Датасет содержит достаточно данных: {len(data)} примеров")
            
            # Загружаем модель и токенизатор
            logger.info("Загрузка модели и токенизатора...")
            model_to_use = model_name or self.config.model_name
            self.model, self.tokenizer = load_model(model_to_use)
            logger.info("Модель и токенизатор загружены")
            
            # Подготавливаем датасет с разделением на train/val
            logger.info("Подготовка датасета с train/val split...")
            if system_prompt:
                logger.info("Заменяем плейсхолдеры системного промпта на реальный промпт")
            
            self.dataset, self.validation_data = prepare_dataset_with_split(data, self.tokenizer, system_prompt)
            self.system_prompt = system_prompt
            logger.info("Датасет подготовлен с разделением на train/validation")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            raise
        
    def objective_function(self, trial) -> float:
        """Целевая функция для байесовской оптимизации"""
        # Получаем параметры от оптимизатора
        lora_params = self.optimizer.suggest_lora_params(trial)
        
        # Переменная для отслеживания PEFT модели
        peft_model = None
        
        # Создаем временную директорию для сохранения адаптера
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                logger.info(f"Trial {trial.number}: Начинаем обучение с параметрами {lora_params}")
                
                # Создаем чистую копию базовой модели для нового trial'а
                # Важно: не модифицируем self.model, а работаем с копией
                current_model = self.model
                
                # Проверяем, есть ли уже PEFT адаптеры в базовой модели
                if hasattr(current_model, 'peft_config') and current_model.peft_config:
                    logger.info("Найдены существующие PEFT адаптеры, создаем чистую копию базовой модели...")
                    # Получаем чистую базовую модель без изменения self.model
                    current_model = safe_unload_peft_adapters(current_model)
                
                # Дообучаем модель
                peft_model = fine_tune_lora(
                    current_model, 
                    self.tokenizer, 
                    self.dataset, 
                    lora_params, 
                    temp_dir,
                    self.config  # Передаем конфигурацию
                )
                
                # Оценка качества через BERTScore на validation set
                bert_score = self.evaluator.quick_evaluate(
                    peft_model, 
                    self.tokenizer, 
                    self.validation_data, 
                    self.system_prompt
                )
                
                logger.info(f"Trial {trial.number}: params={lora_params}, BERTScore={bert_score:.4f}")
                
                # Логируем trial в MLflow
                if self.mlflow_tracker and self.mlflow_tracker.enabled:
                    trial_metrics = {
                        "bert_score": bert_score,
                        "trial_score": bert_score  # Основная метрика для оптимизации
                    }
                    
                    # Добавляем дополнительные метрики если есть
                    if hasattr(peft_model, 'training_logs'):
                        trial_metrics.update(peft_model.training_logs)
                    
                    self.mlflow_tracker.log_bayesian_trial(
                        trial_number=trial.number,
                        trial_params=lora_params,
                        trial_metrics=trial_metrics,
                        model_path=temp_dir if os.path.exists(temp_dir) else None
                    )
                
                return bert_score
                
            except Exception as e:
                logger.error(f"Ошибка в trial {trial.number}: {e}", exc_info=True)
                
                # Логируем неудачный trial в MLflow
                if self.mlflow_tracker and self.mlflow_tracker.enabled:
                    self.mlflow_tracker.log_bayesian_trial(
                        trial_number=trial.number,
                        trial_params=lora_params,
                        trial_metrics={"error": 1.0, "bert_score": 0.0}
                    )
                
                return 0.0
            finally:
                # Критически важно: очищаем PEFT модель после каждого trial
                if peft_model is not None:
                    cleanup_peft_model(peft_model)
                
                # Дополнительная очистка памяти GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Принудительная сборка мусора
                import gc
                gc.collect()
                
                logger.debug(f"Trial {trial.number}: память очищена")
    
    def run_optimization(self, data_path: str, output_dir: str = "./lora_results", model_name: str = None, system_prompt: str = None) -> Dict[str, Any]:
        """Запускает полный цикл оптимизации и дообучения
        
        Args:
            data_path: Путь к файлу с данными
            output_dir: Директория для сохранения результатов
            model_name: Название модели для загрузки
            system_prompt: Системный промпт для замены плейсхолдеров
        """
        logger.info("Начинаем дообучение LoRA с байесовской оптимизацией...")
        
        # Загружаем данные
        self.load_data(data_path, model_name, system_prompt)
        
        # Обновляем метаданные эксперимента
        self.experiment_metadata.update({
            "dataset_size": len(self.dataset),
            "validation_size": len(self.validation_data),
            "model_name": model_name or self.config.model_name,
            "data_path": data_path,
            "output_dir": output_dir,
            "n_trials": self.config.n_trials
        })
        
        # Запускаем байесовскую оптимизацию
        logger.info(f"Запускаем оптимизацию на {self.config.n_trials} попыток...")
        best_params = self.optimizer.optimize(self.objective_function)
        
        # Проверяем, что параметры были найдены
        if not best_params:
            raise RuntimeError(
                "Байесовская оптимизация не нашла лучших параметров. "
                "Возможно, все попытки завершились с ошибкой."
            )
            
        logger.info(f"Лучшие параметры: {best_params}")
        
        # Создаем выходную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        # Обучаем финальную модель с лучшими параметрами
        logger.info("Обучаем финальную модель с лучшими параметрами...")
        final_model = fine_tune_lora(
            self.model,
            self.tokenizer,
            self.dataset,
            best_params,
            output_dir,
            self.config  # Передаем конфигурацию
        )
        
        # Детальная оценка финальной модели через BERTScore + LLM Judge
        logger.info("Проводим детальную оценку финальной модели...")
        detailed_metrics = self.evaluator.detailed_evaluate(
            final_model,
            self.tokenizer, 
            self.validation_data,
            self.system_prompt
        )
        
        logger.info(f"Финальные метрики:")
        logger.info(f"  BERTScore: {detailed_metrics['bert_score']:.4f}")
        
        # Безопасное логирование LLM Judge score
        llm_judge_score = detailed_metrics.get('llm_judge_score')
        if llm_judge_score is not None:
            logger.info(f"  LLM Judge: {llm_judge_score:.2f}/100")
        else:
            logger.info("  LLM Judge: выключен")
            
        logger.info(f"  Combined: {detailed_metrics['combined_score']:.4f}")
        
        # Сохраняем результаты
        results = {
            'best_params': best_params,
            'output_dir': output_dir,
            'n_trials': self.config.n_trials,
            'metrics': detailed_metrics,
            'experiment_metadata': self.experiment_metadata
        }
        
        # Логируем финальную модель в MLflow
        if self.mlflow_tracker and self.mlflow_tracker.enabled:
            optimization_results = {
                **results,
                "best_score": detailed_metrics['combined_score']
            }
            
            self.mlflow_tracker.log_final_model(
                best_params=best_params,
                final_metrics=detailed_metrics,
                model_path=output_dir,
                optimization_results=optimization_results
            )
            
            # Получаем URL для просмотра в MLflow UI
            mlflow_url = self.mlflow_tracker.get_run_url()
            if mlflow_url:
                logger.info(f"MLflow эксперимент доступен по ссылке: {mlflow_url}")
                results['mlflow_url'] = mlflow_url
        
        with open(os.path.join(output_dir, 'optimization_results.json'), 'w') as f:
            # Создаем копию без experiment_metadata для JSON serialization
            json_results = {k: v for k, v in results.items() if k != 'experiment_metadata'}
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Дообучение завершено! Результаты сохранены в {output_dir}")
        return results


if __name__ == "__main__":
    """Точка входа для прямого запуска дообучения LoRA"""
    import argparse
    import sys
    import traceback
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Запуск дообучения LoRA...")
    logger.info(f"Python версия: {sys.version}")
    logger.info(f"Аргументы командной строки: {sys.argv}")
    
    try:
        parser = argparse.ArgumentParser(description="Запуск дообучения LoRA с байесовской оптимизацией")
        parser.add_argument("--data", type=str, required=True, 
                           help="Путь к файлу с данными для обучения (JSONL)")
        parser.add_argument("--output", type=str, default="./lora_results",
                           help="Директория для сохранения результатов")
        parser.add_argument("--trials", type=int, default=5,
                           help="Количество попыток байесовской оптимизации")
        parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                           help="Название модели для дообучения. Популярные варианты: "
                                "Qwen/Qwen3-0.6B, "
                                "microsoft/DialoGPT-medium, "
                                "meta-llama/Llama-2-7b-chat-hf, "
                                "HuggingFaceH4/zephyr-7b-beta. "
                                "По умолчанию: Qwen/Qwen3-0.6B")
        
        logger.info("Парсинг аргументов...")
        args = parser.parse_args()
        logger.info(f"Аргументы получены: data={args.data}, output={args.output}, trials={args.trials}, model={args.model}")
        
        # Проверяем существование файла
        logger.info(f"Проверка файла данных: {args.data}")
        if not os.path.exists(args.data):
            logger.error(f"Файл {args.data} не найден!")
            sys.exit(1)
        else:
            logger.info(f"Файл найден: {args.data}")
        
        # Создаем конфигурацию
        logger.info("Создание конфигурации...")
        config = LoRATuningConfig.from_env()
        config.n_trials = args.trials
        logger.info(f"Конфигурация создана: trials={config.n_trials}")
        
        # Создаем и запускаем тюнер
        logger.info("Создание LoRATuner...")
        tuner = LoRATuner(config)
        logger.info("LoRATuner создан")
        
        logger.info("Запуск оптимизации...")
        results = tuner.run_optimization(args.data, args.output, args.model)
        
        logger.info("Дообучение LoRA успешно завершено!")
        logger.info(f"Лучшие параметры: {results['best_params']}")
        logger.info(f"Результаты сохранены в: {results['output_dir']}")
        
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}", exc_info=True)
        logger.error("Возможно, не установлены необходимые зависимости")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        sys.exit(1) 