# LoRA Fine-tuning Module

## Обзор

Модуль `worker/lora/` отвечает за дообучение LoRA-адаптеров языковых моделей с байесовской оптимизацией гиперпараметров и интеллектуальной оценкой качества.

## Основные компоненты

### 1. `lora_tuning.py`
Основной модуль с классом `LoRATuner` для запуска дообучения.

### 2. `bayesian_optimizer.py` 
Байесовская оптимизация гиперпараметров LoRA с использованием Optuna.

### 3. `evaluation.py`
Система оценки качества моделей через:
- **BERTScore**: Быстрая семантическая оценка
- **LLM Judge**: Продвинутая оценка через дополнительную языковую модель

### 4. `lora_tuning_config.py`
Конфигурация параметров дообучения.

## Новая функциональность: Опциональный LLM Judge

### Что такое LLM Judge?

LLM Judge — это система оценки качества обученной модели с помощью дополнительной языковой модели. Она анализирует качество генерируемых ответов на более высоком уровне, чем простые метрики.

### Преимущества LLM Judge:
- 🎯 **Более точная оценка**: Учитывает контекст и семантику
- 🧠 **Интеллектуальная оценка**: Понимает стиль и соответствие задаче
- 📊 **Комбинированные метрики**: BERTScore (40%) + LLM Judge (60%)

### Недостатки LLM Judge:
- ⏰ **Увеличенное время**: Дополнительные API-вызовы
- 💰 **Повышенная стоимость**: Использование внешних LLM API
- 🔄 **Дополнительная сложность**: Зависимость от внешних сервисов

## Настройка через API

### Запрос с настройками LLM Judge:

```json
{
  "project_id": 123,
  "fine_tuning_params": {
    "use_llm_judge": true,
    "judge_model_id": "gemini",
    "base_model_name": "Qwen/Qwen3-0.6B",
    "n_trials": 20
  }
}
```

### Запрос БЕЗ LLM Judge:

```json
{
  "project_id": 123,
  "fine_tuning_params": {
    "use_llm_judge": false,
    "base_model_name": "Qwen/Qwen3-0.6B",
    "n_trials": 15
  }
}
```

## Настройка через веб-интерфейс

В веб-интерфейсе доступны два варианта запуска fine-tuning:

### 1. 🔥 Настроить Fine-tuning
Открывает модальное окно с опциями:
- ✅ **Включить LLM Judge** (чекбокс)
- 🤖 **Модель для LLM Judge** (выбор из доступных)
- 🎯 **Базовая модель** (HuggingFace название)
- 🔄 **Количество попыток оптимизации** (5-100)

### 2. ⚡ Быстрый запуск
Запускает с настройками по умолчанию:
- LLM Judge: включен
- Модель Judge: по умолчанию
- Попытки: 20

## Логи и мониторинг

### Примеры логов с LLM Judge:
```
INFO - LLM Judge включен с моделью: gemini
INFO - Trial 5: params={'rank': 32, 'lora_alpha': 64}, BERTScore=0.8234
INFO - LLM Judge средняя оценка: 78.50 (на 5 примерах)
INFO - Детальная оценка: BERTScore=0.8234, LLM Judge=78.50, Combined=0.8004
```

### Примеры логов БЕЗ LLM Judge:
```
INFO - LLM Judge выключен, будет использоваться только BERTScore
INFO - Trial 5: params={'rank': 32, 'lora_alpha': 64}, BERTScore=0.8234
INFO - Детальная оценка (без LLM Judge): BERTScore=0.8234
```

## Рекомендации по использованию

### Когда использовать LLM Judge:
- 🎯 **Высокие требования к качеству**
- 💰 **Достаточный бюджет** на API-вызовы
- ⏰ **Нет жестких ограничений по времени**
- 🎨 **Важна оценка стиля и креативности**

### Когда НЕ использовать LLM Judge:
- ⚡ **Быстрое прототипирование**
- 💸 **Ограниченный бюджет**
- 🔒 **Строгие требования приватности**
- 📊 **Простые задачи (классификация, etc.)**

## Конфигурация

### Переменные окружения:
```bash
# Основные настройки LoRA
LORA_MODEL_NAME=Qwen/Qwen3-0.6B
LORA_N_TRIALS=20

# Настройки обучения
LORA_NUM_EPOCHS=3
LORA_BATCH_SIZE=4
LORA_LEARNING_RATE=0.0001
```

### Поддерживаемые модели для LLM Judge:
- `gemini` (Google Gemini)
- `gigachat` (Сбер GigaChat)
- Любые другие, зарегистрированные в `shared.llm`

## Архитектура оценки

```
Validation Data → Model Predictions
                       ↓
               ┌─── BERTScore (быстро)
               │
               └─── LLM Judge (точно, если включен)
                       ↓
               Combined Score (финальная метрика)
```

### Формула комбинированной метрики:
- **С LLM Judge**: `Combined = BERTScore * 0.4 + LLMJudge * 0.6`
- **Без LLM Judge**: `Combined = BERTScore * 1.0`

## Troubleshooting

### Ошибка "LLM Judge модель не найдена":
```python
# Проверьте доступные модели:
from shared.llm import get_model_by_id
model = get_model_by_id("your_model_id")
```

### Высокое время выполнения:
- Уменьшите `n_trials`
- Отключите LLM Judge
- Уменьшите размер validation выборки

### Низкое качество без LLM Judge:
- Увеличьте `n_trials`
- Включите LLM Judge
- Проверьте качество датасета 