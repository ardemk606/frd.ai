"""
Основной клиент для работы с MinIO
"""
import json
import logging
from io import BytesIO
from typing import List, Optional, Union, Dict, Any
from minio import Minio
from minio.error import S3Error
import tempfile

from .models import (
    MinIOConfig, ObjectInfo, UploadResult, DownloadResult, ContentType,
    FileUploadRequest, JSONLUploadRequest, TextUploadRequest
)
from .exceptions import (
    MinIOError, ObjectNotFoundError, UploadError, 
    DownloadError, BucketError, ValidationError
)

logger = logging.getLogger(__name__)


class MinIOClient:
    """Высокоуровневый клиент для работы с MinIO"""
    
    def __init__(self, config: Optional[MinIOConfig] = None):
        """
        Инициализация клиента
        
        Args:
            config: Конфигурация MinIO, если не указана - загружается из env
        """
        self.config = config or MinIOConfig.from_env()
        self._client = Minio(
            self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
            region=self.config.region
        )
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Проверяет существование bucket и создает его при необходимости"""
        try:
            if not self._client.bucket_exists(self.config.bucket_name):
                self._client.make_bucket(self.config.bucket_name, location=self.config.region)
                logger.info(f"Создан bucket: {self.config.bucket_name}")
        except S3Error as e:
            logger.error(f"Ошибка при работе с bucket {self.config.bucket_name}: {e}")
            raise BucketError(f"Ошибка при работе с bucket: {str(e)}") from e
    
    def _prepare_content_for_upload(self, content: Union[str, bytes, BytesIO]) -> tuple[BytesIO, int]:
        """
        Подготавливает контент для загрузки
        
        Args:
            content: Контент для загрузки
            
        Returns:
            Кортеж (BytesIO объект, размер)
        """
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
            return BytesIO(content_bytes), len(content_bytes)
        elif isinstance(content, bytes):
            return BytesIO(content), len(content)
        elif isinstance(content, BytesIO):
            # Сохраняем позицию, получаем размер, возвращаем позицию
            current_pos = content.tell()
            content.seek(0, 2)  # конец файла
            size = content.tell()
            content.seek(current_pos)  # вернуть позицию
            return content, size
        else:
            raise ValidationError(f"Неподдерживаемый тип контента: {type(content)}")
    
    def upload_file(self, request: FileUploadRequest) -> UploadResult:
        """
        Загружает файл в MinIO
        
        Args:
            request: Запрос на загрузку файла
            
        Returns:
            Результат загрузки
        """
        try:
            # Подготавливаем контент
            data, size = self._prepare_content_for_upload(request.content)
            
            # Загружаем в MinIO
            self._client.put_object(
                bucket_name=self.config.bucket_name,
                object_name=request.object_name,
                data=data,
                length=size,
                content_type=request.content_type.value if request.content_type else None,
                metadata=request.metadata
            )
            
            # Создаем информацию об объекте
            object_info = ObjectInfo(
                object_name=request.object_name,
                bucket_name=self.config.bucket_name,
                size=size,
                content_type=request.content_type.value if request.content_type else None
            )
            
            logger.info(f"Файл успешно загружен: {request.object_name}")
            
            return UploadResult(success=True, object_info=object_info)
            
        except S3Error as e:
            error_msg = f"Ошибка S3 при загрузке {request.object_name}: {str(e)}"
            logger.error(error_msg)
            raise UploadError(error_msg) from e
        except Exception as e:
            error_msg = f"Неожиданная ошибка при загрузке {request.object_name}: {str(e)}"
            logger.error(error_msg)
            raise UploadError(error_msg) from e
    
    def upload_text(self, request: TextUploadRequest) -> UploadResult:
        """
        Загружает текстовый файл в MinIO
        
        Args:
            request: Запрос на загрузку текста
            
        Returns:
            Результат загрузки
        """
        file_request = FileUploadRequest(
            object_name=request.object_name,
            content=request.text_content,
            content_type=request.content_type,
            metadata=request.metadata
        )
        return self.upload_file(file_request)
    
    def upload_jsonl(self, request: JSONLUploadRequest) -> UploadResult:
        """
        Загружает JSONL файл в MinIO
        
        Args:
            request: Запрос на загрузку JSONL
            
        Returns:
            Результат загрузки
        """
        # Преобразуем список объектов в JSONL
        jsonl_content = ""
        for obj in request.json_objects:
            jsonl_content += json.dumps(obj, ensure_ascii=False) + '\n'
        
        file_request = FileUploadRequest(
            object_name=request.object_name,
            content=jsonl_content,
            content_type=request.content_type,
            metadata=request.metadata
        )
        return self.upload_file(file_request)
    
    def download_object(self, object_name: str, as_text: bool = True) -> DownloadResult:
        """
        Скачивает объект из MinIO
        
        Args:
            object_name: Имя объекта
            as_text: Если True, возвращает контент как строку, иначе как байты
            
        Returns:
            Результат скачивания
        """
        try:
            # Получаем объект
            response = self._client.get_object(self.config.bucket_name, object_name)
            content_bytes = response.read()
            response.close()
            response.release_conn()
            
            # Преобразуем в нужный формат
            content = content_bytes.decode('utf-8') if as_text else content_bytes
            
            # Получаем информацию об объекте
            object_info = ObjectInfo(
                object_name=object_name,
                bucket_name=self.config.bucket_name,
                size=len(content_bytes)
            )
            
            logger.info(f"Объект успешно скачан: {object_name}")
            
            return DownloadResult(
                success=True,
                content=content,
                object_info=object_info
            )
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                error_msg = f"Объект не найден: {object_name}"
                logger.warning(error_msg)
                raise ObjectNotFoundError(error_msg) from e
            else:
                error_msg = f"Ошибка S3 при скачивании {object_name}: {str(e)}"
                logger.error(error_msg)
                raise DownloadError(error_msg) from e
        except Exception as e:
            error_msg = f"Неожиданная ошибка при скачивании {object_name}: {str(e)}"
            logger.error(error_msg)
            raise DownloadError(error_msg) from e
    
    def download_text(self, object_name: str) -> str:
        """
        Скачивает текстовый объект из MinIO
        
        Args:
            object_name: Имя объекта
            
        Returns:
            Содержимое объекта как строка
            
        Raises:
            ObjectNotFoundError: Если объект не найден
            DownloadError: При ошибке скачивания
        """
        result = self.download_object(object_name, as_text=True)
        return result.content_as_text
    
    def download_jsonl_preview(self, object_name: str, max_lines: int = 5) -> List[Dict[str, Any]]:
        """
        Скачивает превью JSONL файла из MinIO
        
        Args:
            object_name: Имя объекта
            max_lines: Максимальное количество строк для чтения
            
        Returns:
            Список JSON объектов
            
        Raises:
            ObjectNotFoundError: Если объект не найден
            DownloadError: При ошибке скачивания
        """
        try:
            content = self.download_text(object_name)
            
            preview = []
            lines = content.strip().split('\n')[:max_lines]
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    json_obj = json.loads(line)
                    preview.append(json_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка JSON в строке {line_num} объекта {object_name}: {e}")
                    # Продолжаем, а не падаем на одной битой строке
                    
            logger.info(f"Загружен превью JSONL {object_name}: {len(preview)} объектов")
            return preview
            
        except (ObjectNotFoundError, DownloadError):
            raise
        except Exception as e:
            error_msg = f"Ошибка при парсинге JSONL превью {object_name}: {str(e)}"
            logger.error(error_msg)
            raise DownloadError(error_msg) from e
    
    def download_object_to_temp_file(self, object_name: str) -> str:
        """
        Скачивает объект из MinIO во временный файл.

        Args:
            object_name: Имя объекта для скачивания.

        Returns:
            Путь к временному файлу.
            
        Raises:
            ObjectNotFoundError: Если объект не найден.
            DownloadError: При ошибке скачивания.
        """
        try:
            response = self._client.get_object(self.config.bucket_name, object_name)
            
            # Создаем временный файл, который не будет удален после закрытия
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
                for data_chunk in response.stream(32 * 1024):
                    tmp_file.write(data_chunk)
                file_path = tmp_file.name

            response.close()
            response.release_conn()

            logger.info(f"Объект {object_name} скачан во временный файл: {file_path}")
            return file_path

        except S3Error as e:
            if e.code == 'NoSuchKey':
                raise ObjectNotFoundError(f"Объект не найден: {object_name}") from e
            else:
                raise DownloadError(f"Ошибка S3 при скачивании {object_name}: {e}") from e
        except Exception as e:
            raise DownloadError(f"Неожиданная ошибка при скачивании {object_name}: {e}") from e
    
    def list_objects(self, prefix: str = "") -> List[ObjectInfo]:
        """
        Получает список объектов в bucket
        
        Args:
            prefix: Префикс для фильтрации объектов
            
        Returns:
            Список информации об объектах
        """
        try:
            objects = self._client.list_objects(self.config.bucket_name, prefix=prefix)
            
            result = []
            for obj in objects:
                object_info = ObjectInfo(
                    object_name=obj.object_name,
                    bucket_name=self.config.bucket_name,
                    size=obj.size,
                    last_modified=obj.last_modified,
                    etag=obj.etag
                )
                result.append(object_info)
            
            logger.info(f"Найдено {len(result)} объектов с префиксом '{prefix}'")
            return result
            
        except S3Error as e:
            error_msg = f"Ошибка при получении списка объектов: {str(e)}"
            logger.error(error_msg)
            raise MinIOError(error_msg) from e
    
    def delete_object(self, object_name: str) -> bool:
        """
        Удаляет объект из MinIO
        
        Args:
            object_name: Имя объекта
            
        Returns:
            True если успешно удален
        """
        try:
            self._client.remove_object(self.config.bucket_name, object_name)
            logger.info(f"Объект {object_name} успешно удален")
            return True
        except S3Error as e:
            logger.error(f"Ошибка при удалении объекта {object_name}: {e}")
            return False
    
    def object_exists(self, object_name: str) -> bool:
        """
        Проверяет существование объекта
        
        Args:
            object_name: Имя объекта
            
        Returns:
            True если объект существует
        """
        try:
            self._client.stat_object(self.config.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_object_url(self, object_name: str, expires_in_seconds: int = 3600) -> str:
        """
        Получает подписанный URL для объекта
        
        Args:
            object_name: Имя объекта
            expires_in_seconds: Время жизни URL в секундах
            
        Returns:
            Подписанный URL
        """
        try:
            url = self._client.presigned_get_object(
                self.config.bucket_name,
                object_name,
                expires=expires_in_seconds
            )
            return url
        except S3Error as e:
            error_msg = f"Ошибка при создании URL для {object_name}: {str(e)}"
            logger.error(error_msg)
            raise MinIOError(error_msg) from e 