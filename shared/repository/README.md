# Repository Layer

Этот модуль содержит репозиторий для работы с данными, следуя принципам Repository Pattern и best practices.

## Структура

- `base.py` - Базовый абстрактный класс для всех репозиториев
- `dataset_repository.py` - Репозиторий для работы с датасетами
- `models.py` - Модели данных (Dataset, DatasetCreate, DatasetUpdate)
- `exceptions.py` - Кастомные исключения
- `status_service.py` - Сервис для управления статусами
- `dependencies.py` - Фабрики для dependency injection

## Best Practices

### 1. Разделение ответственности
- **Repository** - только операции с БД
- **Service** - бизнес-логика
- **Models** - структуры данных

### 2. Обработка ошибок
- Кастомные исключения для разных типов ошибок
- Логирование всех операций
- Автоматический rollback транзакций

### 3. Транзакции
- Контекстный менеджер `_get_transaction()` для автоматического коммита/отката
- Контекстный менеджер `_get_cursor()` для простых запросов

### 4. Типизация
- Полная типизация всех методов
- Использование dataclasses для моделей
- Type hints для всех параметров

## Использование

```python
from shared.repository import DatasetRepository, DatasetCreate, get_dataset_repository
from shared.repository import get_dataset_repository

# В FastAPI роутерах
@router.post("/dataset")
def create_dataset(
    dataset_data: DatasetCreateRequest,
    repository: DatasetRepository = Depends(get_dataset_repository)
):
    repository = DatasetRepository(db)
    
    dataset_create = DatasetCreate(
        filename=dataset_data.filename,
        object_name=dataset_data.object_name,
        size_bytes=dataset_data.size_bytes,
        system_prompt_object_name=dataset_data.system_prompt_object_name
    )
    
    dataset_id = repository.create(dataset_create)
    return {"id": dataset_id}

# Или с dependency injection
@router.get("/datasets")
def get_datasets(
    repository: DatasetRepository = Depends(
        get_dataset_repository
    )
):
    return repository.get_all()
```

## Методы DatasetRepository

- `get_all()` - Получить все датасеты
- `get_by_id(dataset_id)` - Получить датасет по ID
- `create(dataset_data)` - Создать новый датасет
- `update_status(dataset_id, new_status)` - Обновить статус
- `get_status(dataset_id)` - Получить статус
- `update(dataset_id, update_data)` - Обновить датасет

## Методы DatasetStatusService

- `proceed_to_next_step(dataset_id)` - Перейти к следующему шагу
- `set_status(dataset_id, new_status)` - Установить статус с валидацией
- `get_next_status(current_status)` - Получить следующий статус
- `is_valid_status(status)` - Проверить валидность статуса
- `can_proceed_to_next_step(dataset_id)` - Проверить возможность перехода 