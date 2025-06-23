"""
FastAPI сервер для генерации данных
"""
import logging

from fastapi import FastAPI
import uvicorn

from .routers import health, upload, projects

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Generation API",
    description="API для генерации данных с использованием Google AI",
    version="1.0.0"
)

app.include_router(health.router)           # Системные эндпоинты
app.include_router(upload.router)           # Загрузка датасетов
app.include_router(projects.router)         # Управление проектами


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7777)