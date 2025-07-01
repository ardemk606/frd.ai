import os
import json
import asyncio
import logging
import websockets
from typing import Dict, Any, Optional, AsyncGenerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import uuid

logger = logging.getLogger(__name__)

class InferenceWorker:
    """Воркер для инференса с поддержкой LoRA-адаптеров и streaming"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.base_model = None
        self.tokenizer = None
        self.current_model = None
        self.current_adapter_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # WebSocket соединения
        self.websocket_clients = {}
        
    async def load_base_model(self):
        """Загружает базовую модель"""
        logger.info(f"Загрузка базовой модели: {self.model_name}")
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Загружаем модель
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Устанавливаем текущую модель как базовую
        self.current_model = self.base_model
        
        logger.info(f"Базовая модель загружена на {self.device}")
        
    async def load_lora_adapter(self, adapter_path: str):
        """Загружает LoRA-адаптер поверх базовой модели"""
        if not self.base_model:
            await self.load_base_model()
            
        if self.current_adapter_path == adapter_path:
            logger.info(f"LoRA-адаптер {adapter_path} уже загружен")
            return
            
        logger.info(f"Загрузка LoRA-адаптера: {adapter_path}")
        
        try:
            # Загружаем LoRA-адаптер
            self.current_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path
            )
            self.current_adapter_path = adapter_path
            logger.info(f"LoRA-адаптер загружен: {adapter_path}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки LoRA-адаптера {adapter_path}: {e}")
            # Возвращаемся к базовой модели
            self.current_model = self.base_model
            self.current_adapter_path = None
            raise
            
    async def unload_lora_adapter(self):
        """Выгружает текущий LoRA-адаптер, возвращаясь к базовой модели"""
        if self.current_adapter_path is None:
            logger.info("LoRA-адаптер не загружен")
            return
            
        logger.info(f"Выгрузка LoRA-адаптера: {self.current_adapter_path}")
        self.current_model = self.base_model
        self.current_adapter_path = None
        
        # Очищаем CUDA кеш
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def format_chat_prompt(self, messages: list, system_prompt: str = None) -> str:
        """Форматирует сообщения в промпт для модели"""
        prompt_parts = []
        
        # Добавляем системный промпт если есть
        if system_prompt:
            prompt_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
            
        # Добавляем сообщения
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            
        # Добавляем начало ответа ассистента
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
        
    async def generate_stream(
        self,
        messages: list,
        session_id: str,
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Генерирует стриминговый ответ"""
        
        if not self.current_model:
            await self.load_base_model()
            
        # Форматируем промпт
        prompt = self.format_chat_prompt(messages, system_prompt)
        
        # Токенизируем
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length - max_tokens
        ).to(self.device)
        
        # Создаем streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=30.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Параметры генерации
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Запускаем генерацию в отдельном потоке
        generation_thread = Thread(
            target=self.current_model.generate,
            kwargs=generation_kwargs
        )
        generation_thread.start()
        
        # Стримим токены
        generated_text = ""
        try:
            for new_text in streamer:
                if new_text:
                    generated_text += new_text
                    yield {
                        "type": "token",
                        "content": new_text,
                        "session_id": session_id
                    }
                    
        except Exception as e:
            logger.error(f"Ошибка при генерации: {e}")
            yield {
                "type": "error",
                "content": f"Ошибка генерации: {str(e)}",
                "session_id": session_id
            }
            return
            
        finally:
            generation_thread.join(timeout=5.0)
            
        # Отправляем финальный результат
        yield {
            "type": "done",
            "content": generated_text,
            "session_id": session_id
        }
        
    async def handle_websocket_client(self, websocket):
        """Обработчик WebSocket соединений"""
        client_id = str(uuid.uuid4())
        self.websocket_clients[client_id] = websocket
        
        logger.info(f"Новое WebSocket соединение: {client_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_websocket_message(data, client_id, websocket)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": "Неверный формат JSON"
                    }))
                except Exception as e:
                    logger.error(f"Ошибка обработки сообщения: {e}")
                    await websocket.send(json.dumps({
                        "type": "error", 
                        "content": str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket соединение закрыто: {client_id}")
        finally:
            if client_id in self.websocket_clients:
                del self.websocket_clients[client_id]
                
    async def process_websocket_message(self, data: Dict[str, Any], client_id: str, websocket):
        """Обрабатывает сообщения от WebSocket клиента"""
        message_type = data.get("type")
        
        if message_type == "load_adapter":
            adapter_path = data.get("adapter_path")
            if adapter_path:
                try:
                    await self.load_lora_adapter(adapter_path)
                    await websocket.send(json.dumps({
                        "type": "adapter_loaded",
                        "adapter_path": adapter_path
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": f"Ошибка загрузки адаптера: {str(e)}"
                    }))
            else:
                # Выгружаем адаптер (используем базовую модель)
                await self.unload_lora_adapter()
                await websocket.send(json.dumps({
                    "type": "adapter_unloaded"
                }))
                
        elif message_type == "generate":
            messages = data.get("messages", [])
            session_id = data.get("session_id", client_id)
            system_prompt = data.get("system_prompt")
            max_tokens = data.get("max_tokens", 512)
            temperature = data.get("temperature", 0.7)
            top_k = data.get("top_k", 50)
            top_p = data.get("top_p", 0.9)
            
            # Генерируем стриминговый ответ
            async for response in self.generate_stream(
                messages=messages,
                session_id=session_id,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            ):
                await websocket.send(json.dumps(response))
                
        elif message_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
            
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "content": f"Неизвестный тип сообщения: {message_type}"
            }))


async def main():
    """Основная функция запуска inference worker'а"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    model_name = os.getenv("INFERENCE_MODEL_NAME", "Qwen/Qwen3-0.6B")
    host = os.getenv("INFERENCE_HOST", "0.0.0.0")
    port = int(os.getenv("INFERENCE_PORT", "8765"))
    
    worker = InferenceWorker(model_name)
    
    # Предзагружаем базовую модель
    logger.info("Предзагрузка базовой модели...")
    await worker.load_base_model()
    logger.info("Базовая модель готова к работе")
    
    # Создаем обработчик для websockets.serve
    async def websocket_handler(websocket):
        await worker.handle_websocket_client(websocket)
    
    # Запускаем WebSocket сервер
    logger.info(f"Запуск inference worker на {host}:{port}")
    await websockets.serve(websocket_handler, host, port)
    
    logger.info("Inference worker запущен и готов к работе")
    await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main()) 