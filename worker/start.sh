#!/bin/bash

# Устанавливаем PYTHONPATH
export PYTHONPATH=/app:$PYTHONPATH

# Определяем очереди (по умолчанию CPU очереди)
WORKER_QUEUES=${WORKER_QUEUES:-"celery,cpu_queue"}

echo "Запуск Celery worker с очередями: $WORKER_QUEUES"

# Запускаем Celery worker с указанными очередями
celery -A celery_app worker --loglevel=info --queues=$WORKER_QUEUES 