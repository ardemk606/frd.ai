FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY app.py .

# Открываем порт для Streamlit
EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 