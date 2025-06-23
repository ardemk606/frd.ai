#!/bin/bash

# Устанавливаем PYTHONPATH
export PYTHONPATH=/app:$PYTHONPATH

echo "Запуск Celery worker..."

# Запускаем Celery worker с поддержкой cpu_queue и gpu_queue очередей
celery -A celery_app worker --loglevel=info --queues=celery,cpu_queue,gpu_queue 