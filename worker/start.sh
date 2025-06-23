#!/bin/bash

# Устанавливаем PYTHONPATH
export PYTHONPATH=/app:$PYTHONPATH

echo "Запуск Celery worker..."

# Запускаем только Celery worker
celery -A celery_app worker --loglevel=info 