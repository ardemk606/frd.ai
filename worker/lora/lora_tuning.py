import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, List
import json
import os
import tempfile

# Импорты из того же пакета
try:
    from .bayesian_optimizer import BayesianOptimizer
    from .lora_tuning_config import LoRATuningConfig
except ImportError:
    # Если запускаем напрямую, импортируем абсолютно
    from lora.bayesian_optimizer import BayesianOptimizer
    from lora.lora_tuning_config import LoRATuningConfig


def load_model():
    """Загружает модель Mistral"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Загружаем модель
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, tokenizer

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
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def fine_tune_lora(model, tokenizer, dataset, lora_params: Dict[str, Any], output_dir: str):
    """Запускает LoRA дообучение"""
    # Настраиваем LoRA
    peft_model = setup_lora_model(model, lora_params)
    
    # Настраиваем параметры обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=lora_params['learning_rate'],
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        warmup_steps=100,
        remove_unused_columns=False,
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
        
    def load_data(self, data_path: str) -> None:
        """Загружает данные для обучения"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        print(f"Загружено {len(data)} примеров из {data_path}")
        
        # Загружаем модель и токенизатор
        self.model, self.tokenizer = load_model()
        
        # Подготавливаем датасет
        self.dataset = prepare_dataset(data, self.tokenizer)
        
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
                
                print(f"Trial {trial.number}: params={lora_params}, score={score:.4f}")
                return score
                
            except Exception as e:
                print(f"Ошибка в trial {trial.number}: {e}")
                return 0.0
    
    def run_optimization(self, data_path: str, output_dir: str = "./lora_results") -> Dict[str, Any]:
        """Запускает полный цикл оптимизации и дообучения"""
        print("Начинаем дообучение LoRA с байесовской оптимизацией...")
        
        # Загружаем данные
        self.load_data(data_path)
        
        # Запускаем байесовскую оптимизацию
        print(f"Запускаем оптимизацию на {self.config.n_trials} попыток...")
        best_params = self.optimizer.optimize(self.objective_function)
        
        print(f"Лучшие параметры: {best_params}")
        
        # Создаем выходную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        # Обучаем финальную модель с лучшими параметрами
        print("Обучаем финальную модель с лучшими параметрами...")
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
        
        print(f"Дообучение завершено! Результаты сохранены в {output_dir}")
        return results


if __name__ == "__main__":
    """Точка входа для прямого запуска дообучения LoRA"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск дообучения LoRA с байесовской оптимизацией")
    parser.add_argument("--data", type=str, required=True, 
                       help="Путь к файлу с данными для обучения (JSONL)")
    parser.add_argument("--output", type=str, default="./lora_results",
                       help="Директория для сохранения результатов")
    parser.add_argument("--trials", type=int, default=5,
                       help="Количество попыток байесовской оптимизации")
    
    args = parser.parse_args()
    
    # Создаем конфигурацию
    config = LoRATuningConfig.from_env()
    config.n_trials = args.trials
    
    # Создаем и запускаем тюнер
    tuner = LoRATuner(config)
    
    try:
        results = tuner.run_optimization(args.data, args.output)
        print("✅ Дообучение LoRA успешно завершено!")
        print(f"📊 Лучшие параметры: {results['best_params']}")
        print(f"📁 Результаты сохранены в: {results['output_dir']}")
        
    except Exception as e:
        print(f"❌ Ошибка при дообучении: {e}")
        exit(1) 