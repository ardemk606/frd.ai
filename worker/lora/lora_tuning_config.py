import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class LoRATuningConfig:
    """Конфигурация для LoRA дообучения"""
    
    # Параметры модели
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    max_length: int = 512
    
    # Параметры обучения
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 100
    
    # Параметры байесовской оптимизации
    n_trials: int = 20
    
    # Target modules для LoRA
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    @classmethod
    def from_env(cls) -> 'LoRATuningConfig':
        """Создает конфигурацию из переменных окружения"""
        load_dotenv()
        
        return cls(
            model_name=os.getenv("LORA_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1"),
            max_length=int(os.getenv("LORA_MAX_LENGTH", "512")),
            num_train_epochs=int(os.getenv("LORA_NUM_EPOCHS", "3")),
            per_device_train_batch_size=int(os.getenv("LORA_BATCH_SIZE", "4")),
            gradient_accumulation_steps=int(os.getenv("LORA_GRAD_ACCUM_STEPS", "2")),
            logging_steps=int(os.getenv("LORA_LOGGING_STEPS", "10")),
            save_steps=int(os.getenv("LORA_SAVE_STEPS", "500")),
            eval_steps=int(os.getenv("LORA_EVAL_STEPS", "500")),
            warmup_steps=int(os.getenv("LORA_WARMUP_STEPS", "100")),
            n_trials=int(os.getenv("LORA_N_TRIALS", "20")),
        ) 