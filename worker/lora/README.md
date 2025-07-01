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

## Конфигурация через переменные окружения

```bash
# Основные настройки
LORA_MODEL_NAME=Qwen/Qwen3-0.6B
LORA_N_TRIALS=20

# Опциональные компоненты
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=LoRA_Fine_Tuning
```

## Опциональный MLflow Integration 🔧

### Что такое MLflow?

MLflow — это платформа для управления жизненным циклом ML-экспериментов. Она позволяет:

- 📊 **Отслеживать метрики** каждого trial'а байесовской оптимизации
- 🏷️ **Сохранять параметры** и результаты экспериментов
- 🎯 **Сравнивать результаты** разных запусков
- 💾 **Хранить артефакты** (модели, конфигурации, логи)
- 📈 **Визуализировать прогресс** через веб-интерфейс

### Настройка MLflow (опционально)

#### 1. Локальный запуск MLflow

```bash
# Установка MLflow
pip install mlflow>=2.8.0

# Запуск MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns

# MLflow UI будет доступен по адресу: http://localhost:5000
```

#### 2. Настройка переменных окружения

```bash
# Включение MLflow
export MLFLOW_ENABLED=true

# URL MLflow tracking server
export MLFLOW_TRACKING_URI=http://localhost:5000

# Название эксперимента (опционально)
export MLFLOW_EXPERIMENT_NAME="LoRA_Fine_Tuning"
```

#### 3. Docker Compose конфигурация

Добавьте в `docker-compose.yml`:

```yaml
services:
  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
      - ./mlflow_artifacts:/mlflow_artifacts
    command: >
      bash -c "
        pip install mlflow>=2.8.0 &&
        mlflow server 
        --host 0.0.0.0 
        --port 5000 
        --default-artifact-root /mlflow_artifacts
        --backend-store-uri sqlite:///mlflow.db
      "
    
  worker:
    environment:
      MLFLOW_ENABLED: "true"
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
      MLFLOW_EXPERIMENT_NAME: "LoRA_Fine_Tuning"
    depends_on:
      - mlflow
```

### Что логируется в MLflow?

#### Для каждого Trial байесовской оптимизации:

- **Параметры LoRA**: `rank`, `lora_alpha`, `lora_dropout`, `learning_rate`
- **Метрики**: `bert_score`, `llm_judge_score`, `combined_score`  
- **Теги**: `trial_number`, `optimization_type`, `model_type`
- **Артефакты**: LoRA-адаптер (если сохранён)

#### Для финальной модели:

- **Лучшие параметры** после оптимизации
- **Финальные метрики** (BERTScore + LLM Judge)
- **Результаты оптимизации** (количество trials, лучший score)
- **Финальная модель** и конфигурация

### Просмотр результатов

1. **Откройте MLflow UI**: http://localhost:5000
2. **Выберите эксперимент**: `LoRA_Fine_Tuning`
3. **Сравнивайте trial'ы**: сортировка по метрикам, фильтрация
4. **Просматривайте артефакты**: модели, конфигурации, графики
5. **Анализируйте тренды**: какие параметры дают лучшие результаты

### Отключение MLflow

MLflow **полностью опционален**. Система работает без него:

```bash
# Полное отключение
export MLFLOW_ENABLED=false

# Или просто не устанавливайте переменную
unset MLFLOW_ENABLED
```

При отключении MLflow:
- ✅ Все функции fine-tuning работают нормально
- ✅ Метрики отображаются в логах
- ❌ Нет веб-интерфейса для сравнения экспериментов
- ❌ Нет автоматического сохранения артефактов 

## Решение проблем с памятью и совместимостью 🔧

### Проблемы, которые были решены:

#### 1. **Утечки памяти PEFT между trial'ами**
**Проблема**: PEFT адаптеры накапливались в памяти между trial'ами байесовской оптимизации, вызывая предупреждения:
```
UserWarning: You are trying to modify a model with PEFT for a second time
UserWarning: Already found a `peft_config` attribute in the model
```

**Решение**: 
- Добавлена функция `safe_unload_peft_adapters()` для корректной выгрузки адаптеров
- Функция `cleanup_peft_model()` для полной очистки памяти после каждого trial'а
- Улучшенная логика работы с копиями базовой модели

#### 2. **Deprecated warning для `tokenizer` в Trainer**
**Проблема**: 
```
FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
```

**Решение**:
- Автоматическое определение поддерживаемого API через `inspect.signature()`
- Использование `processing_class` в новых версиях transformers
- Обратная совместимость с `tokenizer` в старых версиях
- Подавление deprecated warnings когда необходимо

#### 3. **Неэффективная очистка GPU памяти**
**Проблема**: Memory leak на GPU между trial'ами

**Решение**:
- Агрессивная очистка GPU памяти с `torch.cuda.empty_cache()` и `torch.cuda.synchronize()`
- Перемещение моделей на CPU перед удалением
- Принудительная сборка мусора с `gc.collect()`

### Новые функции:

```python
def safe_unload_peft_adapters(model):
    """Безопасно выгружает PEFT адаптеры из модели"""
    # Пробует различные методы выгрузки:
    # - unload_and_optionally_merge()
    # - merge_and_unload() 
    # - unload()
    # - disable_adapters()
    # - Прямой доступ к base_model
    
def cleanup_peft_model(peft_model):
    """Полная очистка PEFT модели и освобождение памяти"""
    # - Выгрузка адаптеров
    # - Перемещение на CPU
    # - Удаление ссылок
    # - Очистка GPU памяти
    # - Сборка мусора
```

### Улучшенная архитектура objective_function:

```python
def objective_function(self, trial) -> float:
    # 1. Создаем чистую копию базовой модели (не модифицируем self.model)
    current_model = self.model
    
    # 2. Безопасно выгружаем существующие PEFT адаптеры
    if hasattr(current_model, 'peft_config'):
        current_model = safe_unload_peft_adapters(current_model)
    
    # 3. Обучаем с чистой моделью
    peft_model = fine_tune_lora(current_model, ...)
    
    # 4. Критически важно: полная очистка после trial'а
    finally:
        if peft_model is not None:
            cleanup_peft_model(peft_model)
        
        # Дополнительная очистка памяти
        torch.cuda.empty_cache()
        gc.collect()
```

### Тестирование исправлений:

```bash
# Запуск тестов в worker/lora/
cd worker/lora/
python test_memory_management.py
```

Тест проверяет:
- ✅ Совместимость с API Trainer
- ✅ Корректную очистку памяти между trial'ами  
- ✅ Отсутствие значительного роста использования памяти
- ✅ Работу функций выгрузки PEFT адаптеров

### Результат:

После применения исправлений:
- ❌ **Устранены** предупреждения о множественных PEFT адаптерах
- ❌ **Устранены** deprecated warnings для `tokenizer`
- ✅ **Стабильное** использование памяти между trial'ами
- ✅ **Совместимость** с разными версиями transformers
- ✅ **Надежная** байесовская оптимизация без сбоев

### Мониторинг памяти:

Для контроля памяти в production:

```python
# В логах теперь видно:
logger.debug(f"Trial {trial.number}: память очищена")
logger.info("Выгружаем PEFT адаптеры методом unload_and_optionally_merge()")
logger.debug("PEFT модель очищена из памяти")
```

При отключении MLflow:
- ✅ Все функции fine-tuning работают нормально
- ✅ Метрики отображаются в логах
- ❌ Нет веб-интерфейса для сравнения экспериментов
- ❌ Нет автоматического сохранения артефактов 