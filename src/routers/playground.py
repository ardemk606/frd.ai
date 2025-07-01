from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional
import json
import asyncio
import websockets
import logging
import uuid
from datetime import datetime
from pydantic import BaseModel

from shared.repository.dependencies import create_dataset_repository
from shared.repository.dataset_repository import DatasetRepository
from shared.minio.dependencies import get_minio_client
from shared.minio.client import MinIOClient

logger = logging.getLogger(__name__)

# Модели данных для API
class ChatMessage(BaseModel):
    role: str
    content: str

class InferenceRequest(BaseModel):
    messages: List[ChatMessage]
    adapter_id: Optional[int] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

router = APIRouter(prefix="/playground", tags=["playground"])

# Глобальный менеджер WebSocket соединений
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.inference_ws: Optional[websockets.WebSocketClientProtocol] = None
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket клиент подключен: {client_id}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket клиент отключен: {client_id}")
            
    async def send_to_client(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Ошибка отправки сообщения клиенту {client_id}: {e}")
                
    async def connect_to_inference_worker(self):
        """Подключается к inference worker с retry"""
        inference_host = "inference-worker"  # Имя сервиса в docker-compose
        inference_port = 8765
        
        retry_count = 3
        for attempt in range(retry_count):
            try:
                self.inference_ws = await asyncio.wait_for(
                    websockets.connect(f"ws://{inference_host}:{inference_port}"),
                    timeout=10
                )
                logger.info("Подключение к inference worker установлено")
                return
            except Exception as e:
                if attempt == retry_count - 1:  # Последняя попытка
                    logger.error(f"Не удалось подключиться к inference worker после {retry_count} попыток: {e}")
                    raise
                else:
                    logger.warning(f"Попытка {attempt + 1} подключения к inference worker не удалась: {e}")
                    await asyncio.sleep(1)
            
    async def send_to_inference_worker(self, message: dict):
        """Отправляет сообщение inference worker"""
        if not self.inference_ws:
            await self.connect_to_inference_worker()
            
        try:
            await self.inference_ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения inference worker: {e}")
            # Пробуем переподключиться
            try:
                await self.connect_to_inference_worker()
                await self.inference_ws.send(json.dumps(message))
            except Exception as reconnect_error:
                logger.error(f"Ошибка переподключения к inference worker: {reconnect_error}")
                raise
                
    async def listen_inference_worker(self, client_id: str):
        """Слушает ответы от inference worker и пересылает клиенту"""
        if not self.inference_ws:
            return
            
        try:
            async for message in self.inference_ws:
                data = json.loads(message)
                await self.send_to_client(client_id, data)
                
                # Если это конец генерации, прекращаем слушать
                if data.get("type") == "done" or data.get("type") == "error":
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Соединение с inference worker закрыто")
        except Exception as e:
            logger.error(f"Ошибка при получении сообщений от inference worker: {e}")

# Глобальный экземпляр менеджера
manager = WebSocketManager()


@router.get("/adapters")
async def get_available_lora_adapters():
    try:
        # Создаем клиентов
        repo = create_dataset_repository()
        minio_client = get_minio_client()
        
        # Получаем все датасеты со статусом READY_FOR_DEPLOY или DEPLOYED
        # которые содержат LoRA-адаптеры
        datasets = repo.get_all()
        
        adapters = []
        for dataset in datasets:
            if dataset.status in ['READY_FOR_DEPLOY', 'DEPLOYED'] and dataset.lora_adapter_id:
                # Проверяем, есть ли адаптер в MinIO
                try:
                    # Предполагаем, что адаптеры хранятся в bucket 'lora-adapters'
                    adapter_path = f"lora_results/{dataset.id}"
                    
                    # Проверяем существование адаптера в MinIO  
                    if minio_client.object_exists(f"{adapter_path}/adapter_config.json"):
                        adapters.append({
                            "id": dataset.lora_adapter_id,
                            "name": f"LoRA for Dataset {dataset.id}",
                            "dataset_id": dataset.id,
                            "dataset_name": dataset.filename,
                            "adapter_path": adapter_path,
                            "created_at": dataset.uploaded_at.isoformat(),
                            "status": dataset.status
                        })
                except Exception as e:
                    logger.warning(f"Ошибка проверки адаптера для датасета {dataset.id}: {e}")
                    continue
        
        return {
            "success": True,
            "adapters": adapters,
            "count": len(adapters)
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения списка адаптеров: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка адаптеров: {str(e)}")


@router.get("/models")
async def get_available_models():
    """Получает список доступных базовых моделей"""
    models = [
        {
            "model_id": "Qwen/Qwen3-0.6B",
            "display_name": "Qwen3 0.6B",
            "description": "Легкая модель Qwen3 (0.6B параметров)",
            "is_default": True
        },
        {
            "model_id": "microsoft/DialoGPT-medium", 
            "display_name": "DialoGPT Medium",
            "description": "Microsoft DialoGPT средний размер",
            "is_default": False
        },
        {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "display_name": "Llama 2 7B Chat", 
            "description": "Meta Llama 2 7B Chat модель",
            "is_default": False
        }
    ]
    
    return {
        "success": True,
        "models": models
    }


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket эндпоинт для плейграунда"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Получаем сообщение от клиента
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "load_adapter":
                # Загружаем LoRA-адаптер в inference worker
                adapter_path = message.get("adapter_path")
                await manager.send_to_inference_worker({
                    "type": "load_adapter",
                    "adapter_path": adapter_path
                })
                
                # Слушаем ответ от inference worker
                await manager.listen_inference_worker(client_id)
                
            elif message_type == "generate":
                # Отправляем запрос на генерацию в inference worker
                await manager.send_to_inference_worker(message)
                
                # Слушаем стриминговый ответ от inference worker
                await manager.listen_inference_worker(client_id)
                
            elif message_type == "ping":
                await manager.send_to_client(client_id, {"type": "pong"})
                
            else:
                await manager.send_to_client(client_id, {
                    "type": "error",
                    "content": f"Неизвестный тип сообщения: {message_type}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Ошибка WebSocket соединения для клиента {client_id}: {e}")
        manager.disconnect(client_id)


@router.post("/test-inference")
async def test_inference():
    """Тестовый эндпоинт для проверки соединения с inference worker"""
    try:
        # Пробуем подключиться к inference worker
        await manager.connect_to_inference_worker()
        
        # Отправляем ping
        await manager.send_to_inference_worker({"type": "ping"})
        
        return {
            "success": True,
            "message": "Соединение с inference worker установлено"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/inference/stream")
async def stream_inference(request: InferenceRequest):
    """Стриминговый inference с Server-Sent Events"""
    
    async def generate_sse():
        try:
            # Подключаемся к inference worker
            inference_host = "inference-worker"
            inference_port = 8765
            
            # Формируем сообщение для inference worker
            inference_message = {
                "type": "generate",
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "session_id": str(uuid.uuid4()),
                "system_prompt": request.system_prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p
            }
            
            # Если указан адаптер, загружаем его сначала
            if request.adapter_id:
                try:
                    # Получаем информацию об адаптере
                    repo = create_dataset_repository()
                    datasets = repo.get_all()
                    
                    dataset = next((d for d in datasets if d.lora_adapter_id == request.adapter_id), None)
                    if dataset:
                        adapter_path = f"lora_results/{dataset.id}"
                        
                        # Загружаем адаптер через WebSocket с retry
                        adapter_loaded = False
                        retry_count = 3
                        
                        for attempt in range(retry_count):
                            try:
                                ws = await asyncio.wait_for(
                                    websockets.connect(f"ws://{inference_host}:{inference_port}"),
                                    timeout=10
                                )
                                async with ws:
                                    await ws.send(json.dumps({
                                        "type": "load_adapter",
                                        "adapter_path": adapter_path
                                    }))
                                    
                                    # Ждем подтверждения загрузки
                                    response = await ws.recv()
                                    data = json.loads(response)
                                    
                                    if data.get("type") == "adapter_loaded":
                                        adapter_loaded = True
                                        break
                                    else:
                                        if attempt == retry_count - 1:  # Последняя попытка
                                            error_msg = f"Ошибка загрузки адаптера: {data.get('content', 'Неизвестная ошибка')}"
                                            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                                            return
                                            
                            except Exception as retry_error:
                                if attempt == retry_count - 1:  # Последняя попытка
                                    logger.error(f"Не удалось подключиться к inference worker после {retry_count} попыток: {retry_error}")
                                    error_msg = f"Ошибка подключения к inference worker: {str(retry_error)}"
                                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                                    return
                                else:
                                    logger.warning(f"Попытка {attempt + 1} подключения к inference worker не удалась: {retry_error}")
                                    await asyncio.sleep(1)  # Ждем перед следующей попыткой
                                    
                        if not adapter_loaded:
                            error_msg = "Не удалось загрузить адаптер после нескольких попыток"
                            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                            return
                                
                except Exception as e:
                    logger.error(f"Ошибка загрузки адаптера {request.adapter_id}: {e}")
                    error_msg = f"Ошибка загрузки адаптера: {str(e)}"
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                    return
            
            # Подключаемся к inference worker для генерации с retry
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    ws = await asyncio.wait_for(
                        websockets.connect(f"ws://{inference_host}:{inference_port}"),
                        timeout=10
                    )
                    async with ws:
                        # Отправляем запрос на генерацию
                        await ws.send(json.dumps(inference_message))
                        
                        # Стримим ответы
                        async for message in ws:
                            data = json.loads(message)
                            
                            # Отправляем данные в SSE формате
                            yield f"data: {json.dumps(data)}\n\n"
                            
                            # Если это конец генерации, прекращаем
                            if data.get("type") in ["done", "error"]:
                                break
                    return  # Успешно завершили генерацию
                        
                except Exception as retry_error:
                    if attempt == retry_count - 1:  # Последняя попытка
                        logger.error(f"Не удалось подключиться к inference worker для генерации после {retry_count} попыток: {retry_error}")
                        error_msg = f"Ошибка подключения к inference worker: {str(retry_error)}"
                        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                        return
                    else:
                        logger.warning(f"Попытка {attempt + 1} подключения для генерации не удалась: {retry_error}")
                        await asyncio.sleep(1)  # Ждем перед следующей попыткой
                        
        except websockets.exceptions.ConnectionClosed:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Соединение с inference worker закрыто'})}\n\n"
        except Exception as e:
            logger.error(f"Ошибка при inference: {e}")
            error_msg = f"Ошибка inference: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
    
    return StreamingResponse(generate_sse(), media_type="text/plain") 