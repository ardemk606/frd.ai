FROM python:3.11-slim

WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код worker и shared модули
COPY worker/ .
COPY shared/ ./shared/

# Делаем start.sh исполняемым
RUN chmod +x start.sh

# Запускаем Redis consumer и Celery worker
CMD ["./start.sh"] 