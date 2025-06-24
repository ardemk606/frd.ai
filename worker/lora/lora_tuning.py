import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, List
import json

def load_model():
    """Загружает модель Mistral"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
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