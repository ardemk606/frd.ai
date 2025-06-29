"""
FastAPI сервер для генерации данных
"""
from fastapi import FastAPI, Request
import uvicorn

from .routers import health, upload, projects, validation
from shared.logging_config import setup_json_logging, get_logger

setup_json_logging("frd-ai-api")
logger = get_logger(__name__)

app = FastAPI(
    title="Data Generation API",
    description="API для генерации данных с использованием Google AI",
    version="1.0.0"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования HTTP запросов"""
    import uuid
    request_id = str(uuid.uuid4())
    
    # Логируем входящий запрос
    logger.info(
        "HTTP Request",
        extra={
            'extra_data': {
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'headers': dict(request.headers),
                'client_ip': request.client.host
            }
        }
    )
    
    response = await call_next(request)
    
    # Логируем ответ
    logger.info(
        "HTTP Response",
        extra={
            'extra_data': {
                'request_id': request_id,
                'status_code': response.status_code,
                'method': request.method,
                'url': str(request.url)
            }
        }
    )
    
    return response


app.include_router(health.router)           # Системные эндпоинты
app.include_router(upload.router)           # Загрузка датасетов
app.include_router(projects.router)         # Управление проектами
app.include_router(validation.router)       # Валидация датасетов


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7777)