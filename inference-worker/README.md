# Inference Worker

## Назначение (Purpose)

Модуль `inference-worker` — это специализированный сервис для запуска inference языковых моделей с поддержкой LoRA-адаптеров и стримингового вывода. Он предназначен для работы в плейграунде, позволяя пользователям тестировать обученные LoRA-адаптеры в режиме реального времени.

## Основные возможности (Features)

- **Загрузка базовых LLM моделей** (по умолчанию Qwen/Qwen3-0.6B)
- **Поддержка LoRA-адаптеров** с возможностью динамической загрузки/выгрузки
- **WebSocket API** для стримингового inference
- **Настраиваемые параметры генерации** (temperature, top_k, top_p, max_tokens)
- **Чат-формат промптов** с поддержкой системных сообщений
- **Автоматическое управление памятью CUDA**

## Архитектура (Architecture)

Inference Worker построен как асинхронный WebSocket сервер, который:

1. **Предзагружает базовую модель** при запуске
2. **Принимает WebSocket соединения** от FastAPI backend
3. **Обрабатывает команды** для загрузки LoRA-адаптеров
4. **Выполняет inference** с настраиваемыми параметрами
5. **Стримит результаты** по токенам обратно клиенту

## WebSocket API

### Типы сообщений

#### 1. Загрузка LoRA-адаптера
```json
{
  "type": "load_adapter",
  "adapter_path": "/path/to/lora/adapter"
}
```

Ответ:
```json
{
  "type": "adapter_loaded", 
  "adapter_path": "/path/to/lora/adapter"
}
```

#### 2. Выгрузка LoRA-адаптера
```json
{
  "type": "load_adapter",
  "adapter_path": null
}
```

Ответ:
```json
{
  "type": "adapter_unloaded"
}
```

#### 3. Генерация текста
```json
{
  "type": "generate",
  "messages": [
    {"role": "user", "content": "Привет!"}
  ],
  "session_id": "unique-session-id",
  "system_prompt": "Ты полезный ассистент",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9
}
```

Стриминговые ответы:
```json
{
  "type": "token",
  "content": "Привет",
  "session_id": "unique-session-id"
}
```

Финальный ответ:
```json
{
  "type": "done",
  "content": "Полный сгенерированный текст",
  "session_id": "unique-session-id"
}
```

#### 4. Ping/Pong
```json
{"type": "ping"}
```

Ответ:
```json
{"type": "pong"}
```

## Конфигурация (Configuration)

Inference Worker настраивается через переменные окружения:

- `INFERENCE_MODEL_NAME`: Название модели HuggingFace (по умолчанию: "Qwen/Qwen3-0.6B")
- `INFERENCE_HOST`: Хост для WebSocket сервера (по умолчанию: "0.0.0.0")
- `INFERENCE_PORT`: Порт для WebSocket сервера (по умолчанию: 8765)
- `TRANSFORMERS_CACHE`: Директория для кеша моделей
- `HF_HOME`: Домашняя директория HuggingFace

## Запуск (Running)

### 1. Локально (для разработки)

```bash
cd inference-worker
pip install -r requirements.txt
python inference_worker.py
```

### 2. С помощью Docker

```bash
# Собрать и запустить inference worker
docker-compose up --build inference-worker

# Или запустить весь стек
docker-compose up -d
```

## Интеграция с системой (System Integration)

Inference Worker интегрируется с основной системой через:

1. **FastAPI Backend** - WebSocket прокси между фронтендом и inference worker
2. **MinIO** - для загрузки LoRA-адаптеров
3. **Frontend (Streamlit)** - пользовательский интерфейс плейграунда