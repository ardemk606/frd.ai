FROM python:3.11-slim

WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/
COPY shared/ ./shared/
COPY data/ ./data/
COPY mock/ ./mock/

# Экспортируем порт
EXPOSE 7777

# Запускаем приложение
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7777"] 