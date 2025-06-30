FROM python:3.11-slim

WORKDIR /app

# Устанавливаем uv, чтобы соответствовать другим воркерам
# и ускорить установку зависимостей
RUN pip install --no-cache-dir uv

# Копируем файлы приложения и зависимостей.
# Важно: указываем полный путь от корня контекста (который теперь '.'),
# чтобы взять файлы именно из папки frontend.
COPY frontend/requirements.txt ./
COPY frontend/app.py ./

# Устанавливаем зависимости из ПРАВИЛЬНОГО requirements.txt с помощью uv
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Открываем порт для Streamlit
EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]