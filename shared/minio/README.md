# MinIO API

Высокоуровневое API для работы с MinIO объектным хранилищем, следующее всем best practices.

## Структура

- `client.py` - Основной клиент MinIO с базовыми операциями
- `models.py` - Типизированные модели данных 
- `exceptions.py` - Кастомные исключения
- `services.py` - Специализированные сервисы для конкретных задач
- `dependencies.py` - Dependency injection для FastAPI

## Best Practices реализованные

### 1. Типизация
- Полная типизация с dataclasses
- Enum для content types
- Type hints для всех методов

### 2. Скрытие сложности
- Автоматическое преобразование str/bytes → BytesIO
- Автоматическое определение content_type
- Умная обработка ошибок с fallback

### 3. Специализированные API
- `DatasetService` - для работы с датасетами
- `PromptService` - для системных промптов  
- `ProjectStorageService` - объединенные операции

### 4. Обработка ошибок
- Кастомные исключения для разных типов ошибок
- Детальное логирование
- Graceful fallback при ошибках

## Использование

### Базовый клиент

```python
from shared.minio import MinIOClient, FileUploadRequest, ContentType

client = MinIOClient()

# Загрузка файла
request = FileUploadRequest(
    object_name="test/file.txt",
    content="Hello World",
    content_type=ContentType.TEXT_PLAIN
)
result = client.upload_file(request)

# Скачивание файла
content = client.download_text("test/file.txt")

# JSONL превью
preview = client.download_jsonl_preview("datasets/data.jsonl", max_lines=5)
```

### Специализированные сервисы

```python
from shared.minio import get_minio_client, ProjectStorageService

client = get_minio_client()
storage = ProjectStorageService(client)

# Загрузка файлов проекта
dataset_result, prompt_result = storage.upload_project_files(
    dataset_file=uploaded_file,
    system_prompt="You are a helpful assistant"
)

# Получение деталей проекта
dataset_preview, system_prompt = storage.get_project_details(
    dataset_object_name="datasets/20241201_120000_data.jsonl",
    prompt_object_name="prompts/20241201_120000_system_prompt.txt"
)
```

### В FastAPI роутерах

```python
from fastapi import Depends
from shared.minio import create_minio_client_dependency, ProjectStorageService

@router.post("/upload")
def upload_project(
    file: UploadFile = File(...),
    system_prompt: str = Form(...),
    minio_client = Depends(create_minio_client_dependency)
):
    storage = ProjectStorageService(minio_client)
    dataset_result, prompt_result = storage.upload_project_files(file, system_prompt)
    
    return {
        "dataset_object_name": dataset_result.object_name,
        "prompt_object_name": prompt_result.object_name
    }

@router.get("/project/{project_id}/details")
def get_project_details(
    dataset_object_name: str,
    prompt_object_name: str,
    minio_client = Depends(create_minio_client_dependency)
):
    storage = ProjectStorageService(minio_client)
    preview, prompt = storage.get_project_details(dataset_object_name, prompt_object_name)
    
    return {
        "dataset_preview": preview,
        "system_prompt": prompt
    }
```

## Преимущества перед прямым использованием MinIO

### ❌ Было (прямые вызовы):
```python
# Ручная работа с IO
dataset_data = BytesIO(dataset_content)
dataset_size = len(dataset_content)

saver.client.put_object(
    bucket_name=saver.config.bucket_name,
    object_name=dataset_object_name,
    data=dataset_data,
    length=dataset_size,
    content_type='application/jsonl'
)

# Ручной парсинг JSONL
response = saver.client.get_object(bucket_name, object_name)
content = response.read().decode('utf-8')
lines = content.strip().split('\n')[:5]
for line in lines:
    try:
        json_obj = json.loads(line)
        preview.append(json_obj)
    except json.JSONDecodeError:
        # handle error
```

### ✅ Стало (высокоуровневое API):
```python
# Простая загрузка
request = TextUploadRequest(
    object_name="prompts/system.txt",
    text_content=system_prompt
)
result = client.upload_text(request)

# Умный JSONL превью
preview = client.download_jsonl_preview("datasets/data.jsonl", max_lines=5)
```

## Методы MinIOClient

### Загрузка
- `upload_file(request)` - универсальная загрузка
- `upload_text(request)` - загрузка текста
- `upload_jsonl(request)` - загрузка JSONL

### Скачивание  
- `download_object(object_name, as_text=True)` - универсальное скачивание
- `download_text(object_name)` - скачивание как текст
- `download_jsonl_preview(object_name, max_lines=5)` - превью JSONL

### Утилиты
- `list_objects(prefix="")` - список объектов
- `delete_object(object_name)` - удаление
- `object_exists(object_name)` - проверка существования
- `get_object_url(object_name, expires_in_seconds=3600)` - подписанный URL

## Методы сервисов

### DatasetService
- `upload_dataset_file(file, timestamp=None)` - загрузка датасета
- `get_dataset_preview(object_name, max_lines=5)` - превью датасета

### PromptService  
- `upload_system_prompt(prompt_text, timestamp=None)` - загрузка промпта
- `get_system_prompt(object_name)` - получение промпта

### ProjectStorageService
- `upload_project_files(dataset_file, system_prompt)` - загрузка файлов проекта
- `get_project_details(dataset_object_name, prompt_object_name)` - детали проекта 