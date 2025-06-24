import logging
import os
from typing import List, Dict, Any

import google.generativeai as genai

from shared.minio import MinIOClient
from shared.minio.dependencies import get_minio_client

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Класс для генерации ответов через Google Gemini API"""

    def __init__(self, system_prompt_path: str, user_prompt_path: str):
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel('gemini-pro')
        self.minio_client = get_minio_client()

    def _load_prompt_from_minio(self, bucket: str, object_name: str) -> str:
        """Загружает промпт из MinIO"""
        try:
            response = self.minio_client.get_object(bucket, object_name)
            return response.data.decode('utf-8')
        except Exception as e:
            logger.error(f"Error loading prompt '{object_name}' from bucket '{bucket}': {e}")
            raise

    def generate_response(self, json_data: str) -> str:
        """Генерирует ответ на основе JSON данных"""
        try:
            # Получаем системный промпт из MinIO
            system_prompt = self._load_prompt_from_minio(
                bucket=self.minio_client.bucket,
                object_name=self.system_prompt_path
            )
            logger.info(f"System prompt '{self.system_prompt_path}' loaded successfully from bucket '{self.minio_client.bucket}'.")

            # Получаем пользовательский промпт из локального файла в контейнере
            with open(self.user_prompt_path, 'r', encoding='utf-8') as f:
                user_prompt_template = f.read()
            logger.info(f"User prompt '{self.user_prompt_path}' loaded successfully from local path.")

            # Формируем итоговый промпт
            prompt = f"{system_prompt}\\n\\n{user_prompt_template.format(json_data=json_data)}"
            
            logger.debug(f"Generated prompt: {prompt}")
            response = self.model.generate_content(prompt)
            logger.debug(f"Raw response from API: {response.text}")
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise