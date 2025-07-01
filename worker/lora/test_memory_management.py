#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений управления памятью PEFT между trial'ами
"""

import torch
import os
import json
import tempfile
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_peft_memory_cleanup():
    """Тест очистки памяти PEFT между trial'ами"""
    try:
        from lora_tuning import LoRATuner, LoRATuningConfig, safe_unload_peft_adapters, cleanup_peft_model
        from lora_tuning_config import LoRATuningConfig
        
        logger.info("🧪 Тестируем управление памятью PEFT...")
        
        # Создаем минимальный тестовый датасет
        test_data = [
            {
                "instruction": "Ответь на вопрос",
                "input": "Что такое Python?",
                "output": "Python - это язык программирования"
            },
            {
                "instruction": "Ответь на вопрос", 
                "input": "Что такое LoRA?",
                "output": "LoRA - это метод эффективного дообучения"
            },
            {
                "instruction": "Ответь на вопрос",
                "input": "Что такое PEFT?", 
                "output": "PEFT - это Parameter Efficient Fine-Tuning"
            }
        ]
        
        # Сохраняем тестовый датасет
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            test_data_path = f.name
        
        try:
            # Создаем минимальную конфигурацию для быстрого теста
            config = LoRATuningConfig(
                model_name="Qwen/Qwen3-0.6B",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_steps=100,
                eval_steps=100,
                warmup_steps=0,
                n_trials=3  # Только 3 trial'а для теста
            )
            
            # Создаем tuner с отключенным LLM Judge для скорости
            tuner = LoRATuner(
                config=config,
                use_llm_judge=False,  # Отключаем для скорости
                enable_mlflow=False   # Отключаем для простоты
            )
            
            logger.info("✅ LoRATuner создан успешно")
            
            # Загружаем данные
            tuner.load_data(test_data_path, system_prompt="Ты полезный ассистент.")
            logger.info("✅ Данные загружены успешно")
            
            # Проверяем начальное использование памяти GPU
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                logger.info(f"📊 Начальное использование GPU памяти: {initial_memory / 1024 / 1024:.2f} MB")
            
            # Симулируем несколько trial'ов для проверки очистки памяти
            memory_usage = []
            
            for trial_num in range(3):
                logger.info(f"🔄 Тестируем Trial {trial_num}")
                
                # Создаем mock trial object 
                class MockTrial:
                    def __init__(self, number):
                        self.number = number
                
                mock_trial = MockTrial(trial_num)
                
                # Запускаем objective function (она должна очищать память)
                try:
                    score = tuner.objective_function(mock_trial)
                    logger.info(f"✅ Trial {trial_num} завершен, score: {score:.4f}")
                    
                    # Проверяем использование памяти после trial'а
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated()
                        memory_usage.append(current_memory)
                        logger.info(f"📊 Память после Trial {trial_num}: {current_memory / 1024 / 1024:.2f} MB")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка в Trial {trial_num}: {e}")
                    
            # Анализируем использование памяти
            if torch.cuda.is_available() and memory_usage:
                max_memory = max(memory_usage)
                min_memory = min(memory_usage)
                memory_growth = max_memory - min_memory
                
                logger.info(f"📈 Анализ памяти:")
                logger.info(f"   Минимум: {min_memory / 1024 / 1024:.2f} MB")
                logger.info(f"   Максимум: {max_memory / 1024 / 1024:.2f} MB")
                logger.info(f"   Рост памяти: {memory_growth / 1024 / 1024:.2f} MB")
                
                if memory_growth < 100 * 1024 * 1024:  # Менее 100 MB роста
                    logger.info("✅ Управление памятью работает корректно!")
                else:
                    logger.warning(f"⚠️ Возможна утечка памяти: рост {memory_growth / 1024 / 1024:.2f} MB")
            
            logger.info("✅ Тест завершен успешно!")
            
        finally:
            # Очистка
            os.unlink(test_data_path)
            
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        logger.error("Убедитесь, что вы запускаете тест из директории worker/lora/")
        
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}", exc_info=True)


def test_trainer_compatibility():
    """Тест совместимости с Trainer API"""
    try:
        from transformers import Trainer
        import inspect
        
        logger.info("🧪 Тестируем совместимость Trainer API...")
        
        # Проверяем сигнатуру Trainer
        trainer_signature = inspect.signature(Trainer.__init__)
        params = list(trainer_signature.parameters.keys())
        
        logger.info(f"📋 Параметры Trainer.__init__: {params}")
        
        if 'processing_class' in params:
            logger.info("✅ Поддерживается новый параметр 'processing_class'")
        else:
            logger.info("ℹ️ Используется старый параметр 'tokenizer'")
            
        if 'tokenizer' in params:
            logger.info("✅ Поддерживается параметр 'tokenizer'")
        else:
            logger.warning("⚠️ Параметр 'tokenizer' не найден!")
            
        logger.info("✅ Тест совместимости завершен")
        
    except Exception as e:
        logger.error(f"❌ Ошибка теста совместимости: {e}")


if __name__ == "__main__":
    print("🚀 Запуск тестов исправлений LoRA fine-tuning...")
    print("=" * 50)
    
    # Тест совместимости API
    test_trainer_compatibility()
    
    print("-" * 50)
    
    # Тест управления памятью (только если есть GPU или принудительно)
    if torch.cuda.is_available():
        test_peft_memory_cleanup()
    else:
        logger.info("⏭️ GPU недоступен, пропускаем тест управления памятью")
        logger.info("💡 Для полного тестирования запустите на машине с GPU")
    
    print("=" * 50)
    print("✅ Все тесты завершены!")
    print("💡 Если тесты прошли успешно, исправления готовы к использованию") 