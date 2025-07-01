FROM python:3.11-slim AS base

# Установка системных зависимостей с retry-логикой
RUN apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --fix-missing \
    gcc \
    g++ \
    curl \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Установка UV
RUN pip install uv

WORKDIR /app

# =============================================================================
# STAGE 2: Кеширование Python зависимостей с UV
# =============================================================================
FROM base AS dependencies

COPY requirements.txt ./requirements.txt

RUN uv venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu118


# Установка остальных зависимостей (исключая torch чтобы избежать конфликтов)
RUN grep -v "^torch" requirements.txt > requirements_clean.txt && \
    uv pip install -r requirements_clean.txt

# =============================================================================
# STAGE 3: Финальный образ с кодом
# =============================================================================
FROM dependencies AS final

# Создаем структуру директорий
RUN mkdir -p /app/adapters /app/cache

# Копируем код приложения
COPY inference_worker.py .

# Переменные окружения для оптимизации
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Убеждаемся что виртуальное окружение активно
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Переменные окружения по умолчанию для inference worker
ENV INFERENCE_MODEL_NAME="Qwen/Qwen3-0.6B"
ENV INFERENCE_HOST="0.0.0.0"
ENV INFERENCE_PORT="8765"
ENV TRANSFORMERS_CACHE="/app/cache"
ENV HF_HOME="/app/cache"

# Экспортируем порт
EXPOSE 8765

# Запускаем inference worker
CMD ["python", "inference_worker.py"] 