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
except ImportError:
    # Если запускаем напрямую, импортируем абсолютно
    from lora.bayesian_optimizer import BayesianOptimizer
    from lora.lora_tuning_config import LoRATuningConfig

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
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_params['rank'],
        lora_alpha=lora_params['lora_alpha'],
        lora_dropout=lora_params['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    peft_model = get_peft_model(model, lora_config)
    return peft_model

def prepare_dataset(data: List[Dict[str, str]], tokenizer):
    """Подготавливает датасет для обучения"""
    from datasets import Dataset
    
    # Подготавливаем тексты в формате для causal LM
    texts = []
    for item in data:
        text = f"<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
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

def fine_tune_lora(model, tokenizer, dataset, lora_params: Dict[str, Any], output_dir: str):
    """Запускает LoRA дообучение"""
    # Настраиваем LoRA
    peft_model = setup_lora_model(model, lora_params)
    
    # Настраиваем параметры обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Уменьшил batch size для стабильности
        gradient_accumulation_steps=4,  # Увеличил grad accumulation
        learning_rate=lora_params['learning_rate'],
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        warmup_steps=100,
        remove_unused_columns=False,
        dataloader_drop_last=True,  # Избегаем проблем с разными размерами батчей
        fp16=True,  # Используем mixed precision для экономии памяти
        label_names=["labels"],  # Явно указываем колонку с ответами
    )
    
    # Создаем тренер
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


class LoRATuner:
    """Основной класс для дообучения LoRA с байесовской оптимизацией"""
    
    def __init__(self, config: LoRATuningConfig = None):
        """
        Args:
            config: Конфигурация для дообучения
        """
        self.config = config or LoRATuningConfig.from_env()
        self.optimizer = BayesianOptimizer(n_trials=self.config.n_trials)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def load_data(self, data_path: str, model_name: str = None) -> None:
        """Загружает данные для обучения
        
        Args:
            data_path: Путь к файлу с данными
            model_name: Название модели для загрузки
        """
        logger.info(f"Загрузка данных из {data_path}...")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            logger.info(f"Загружено {len(data)} примеров из {data_path}")
            
            # Загружаем модель и токенизатор
            logger.info("Загрузка модели и токенизатора...")
            model_to_use = model_name or self.config.model_name
            self.model, self.tokenizer = load_model(model_to_use)
            logger.info("Модель и токенизатор загружены")
            
            # Подготавливаем датасет
            logger.info("Подготовка датасета...")
            self.dataset = prepare_dataset(data, self.tokenizer)
            logger.info("Датасет подготовлен")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            raise
        
    def objective_function(self, trial) -> float:
        """Целевая функция для байесовской оптимизации"""
        # Получаем параметры от оптимизатора
        lora_params = self.optimizer.suggest_lora_params(trial)
        
        # Создаем временную директорию для сохранения адаптера
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Дообучаем модель
                fine_tune_lora(
                    self.model, 
                    self.tokenizer, 
                    self.dataset, 
                    lora_params, 
                    temp_dir
                )
                
                # Здесь должна быть оценка качества модели
                # Пока возвращаем случайную метрику для демонстрации
                import random
                score = random.uniform(0.7, 0.95)
                
                logger.info(f"Trial {trial.number}: params={lora_params}, score={score:.4f}")
                return score
                
            except Exception as e:
                logger.error(f"Ошибка в trial {trial.number}: {e}", exc_info=True)
                return 0.0
            finally:
                # Освобождаем память CUDA после каждого trial
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def run_optimization(self, data_path: str, output_dir: str = "./lora_results", model_name: str = None) -> Dict[str, Any]:
        """Запускает полный цикл оптимизации и дообучения
        
        Args:
            data_path: Путь к файлу с данными
            output_dir: Директория для сохранения результатов
            model_name: Название модели для загрузки
        """
        logger.info("Начинаем дообучение LoRA с байесовской оптимизацией...")
        
        # Загружаем данные
        self.load_data(data_path, model_name)
        
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
            output_dir
        )
        
        # Сохраняем результаты
        results = {
            'best_params': best_params,
            'output_dir': output_dir,
            'n_trials': self.config.n_trials
        }
        
        with open(os.path.join(output_dir, 'optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
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