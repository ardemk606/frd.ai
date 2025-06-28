# =============================================================================
# STAGE 1: Базовый образ с системными зависимостями
# =============================================================================
FROM python:3.11-slim as base

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =============================================================================
# STAGE 2: Кеширование Python зависимостей
# =============================================================================
FROM base as dependencies

# КРИТИЧНО: Сначала копируем и устанавливаем только requirements.txt
# Это позволяет Docker кешировать слой с зависимостями
COPY worker/requirements.txt ./requirements.txt

# Устанавливаем Python зависимости (включая PyTorch)
# Этот слой будет закеширован и не будет переустанавливаться
# пока не изменится requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# STAGE 3: Финальный образ с кодом
# =============================================================================
FROM dependencies as final

# Создаем необходимые директории
RUN mkdir -p shared data mock

# Копируем код ПОСЛЕ установки зависимостей
# Изменения в коде не будут инвалидировать кеш зависимостей
COPY worker/ .
COPY shared/ ./shared/
COPY data/ ./data/
COPY mock/ ./mock/

# Делаем start.sh исполняемым
RUN chmod +x start.sh

# Переменные окружения для оптимизации
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Запускаем Redis consumer и Celery worker
CMD ["./start.sh"] 
