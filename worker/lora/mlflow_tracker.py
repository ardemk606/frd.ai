"""
Опциональная интеграция с MLflow для отслеживания экспериментов LoRA fine-tuning
"""
import os
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Опциональный импорт MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
    logger.info("MLflow доступен для отслеживания экспериментов")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow не установлен - отслеживание экспериментов отключено")


class MLflowTracker:
    """Трекер экспериментов MLflow с опциональной интеграцией"""
    
    def __init__(self, enabled: bool = None):
        """
        Args:
            enabled: Принудительно включить/выключить MLflow (если None, читается из env)
        """
        self.enabled = self._should_enable(enabled)
        self.experiment_name = None
        self.run_id = None
        
        if self.enabled:
            self._setup_mlflow()
        else:
            logger.info("MLflow отключен - метрики не будут логироваться")
    
    def _should_enable(self, forced_enabled: bool = None) -> bool:
        """Определяет, должен ли быть включен MLflow"""
        if forced_enabled is not None:
            return forced_enabled and MLFLOW_AVAILABLE
        
        # Читаем из переменных окружения
        env_enabled = os.getenv("MLFLOW_ENABLED", "false").lower() in ["true", "1", "yes"]
        return env_enabled and MLFLOW_AVAILABLE
    
    def _setup_mlflow(self):
        """Настройка MLflow"""
        try:
            # Настройка tracking URI
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
            
            # Создание или получение эксперимента
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "LoRA_Fine_Tuning")
            
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Создан новый эксперимент MLflow: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Используется существующий эксперимент MLflow: {experiment_name}")
                
                mlflow.set_experiment(experiment_name)
                self.experiment_name = experiment_name
                
            except Exception as e:
                logger.warning(f"Не удалось настроить эксперимент MLflow: {e}")
                self.enabled = False
                
        except Exception as e:
            logger.error(f"Ошибка настройки MLflow: {e}")
            self.enabled = False
    
    @contextmanager
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Контекстный менеджер для MLflow run"""
        if not self.enabled:
            yield None
            return
        
        try:
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                self.run_id = run.info.run_id
                logger.info(f"Запущен MLflow run: {run.info.run_id}")
                yield run
        except Exception as e:
            logger.error(f"Ошибка MLflow run: {e}")
            yield None
        finally:
            self.run_id = None
    
    def log_params(self, params: Dict[str, Any]):
        """Логирует параметры"""
        if not self.enabled:
            return
        
        try:
            # Конвертируем все параметры в строки/числа для MLflow
            clean_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_params[key] = value
                else:
                    clean_params[key] = str(value)
            
            mlflow.log_params(clean_params)
            logger.debug(f"Параметры залогированы в MLflow: {list(clean_params.keys())}")
        except Exception as e:
            logger.warning(f"Не удалось залогировать параметры в MLflow: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Логирует метрики"""
        if not self.enabled:
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, bool)):
                    mlflow.log_metric(key, float(value), step=step)
            
            logger.debug(f"Метрики залогированы в MLflow: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"Не удалось залогировать метрики в MLflow: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Логирует артефакт"""
        if not self.enabled:
            return
        
        try:
            if os.path.exists(artifact_path):
                if artifact_name:
                    # Логируем файл с новым именем
                    mlflow.log_artifact(artifact_path, artifact_path=artifact_name)
                else:
                    mlflow.log_artifact(artifact_path)
                logger.debug(f"Артефакт залогирован в MLflow: {artifact_path}")
            else:
                logger.warning(f"Артефакт не найден: {artifact_path}")
        except Exception as e:
            logger.warning(f"Не удалось залогировать артефакт в MLflow: {e}")
    
    def log_model(self, model_path: str, model_name: str = "lora_adapter"):
        """Логирует модель"""
        if not self.enabled:
            return
        
        try:
            if os.path.exists(model_path):
                # Логируем как артефакт (для LoRA адаптеров)
                mlflow.log_artifacts(model_path, artifact_path=model_name)
                logger.info(f"Модель залогирована в MLflow: {model_path}")
            else:
                logger.warning(f"Модель не найдена: {model_path}")
        except Exception as e:
            logger.warning(f"Не удалось залогировать модель в MLflow: {e}")
    
    def set_tag(self, key: str, value: str):
        """Устанавливает тег для текущего run"""
        if not self.enabled:
            return
        
        try:
            mlflow.set_tag(key, value)
            logger.debug(f"Тег установлен в MLflow: {key}={value}")
        except Exception as e:
            logger.warning(f"Не удалось установить тег в MLflow: {e}")
    
    def get_run_url(self) -> Optional[str]:
        """Возвращает URL текущего run в MLflow UI"""
        if not self.enabled or not self.run_id:
            return None
        
        try:
            tracking_uri = mlflow.get_tracking_uri()
            return f"{tracking_uri}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{self.run_id}"
        except Exception as e:
            logger.warning(f"Не удалось получить URL run: {e}")
            return None
    
    def log_bayesian_trial(self, trial_number: int, trial_params: Dict[str, Any], 
                          trial_metrics: Dict[str, float], model_path: str = None):
        """Логирует один trial байесовской оптимизации"""
        if not self.enabled:
            return
        
        run_name = f"trial_{trial_number:03d}"
        tags = {
            "trial_number": str(trial_number),
            "optimization_type": "bayesian",
            "model_type": "lora"
        }
        
        with self.start_run(run_name=run_name, tags=tags):
            # Логируем параметры trial'а
            self.log_params(trial_params)
            
            # Логируем метрики
            self.log_metrics(trial_metrics)
            
            # Логируем модель если указан путь
            if model_path and os.path.exists(model_path):
                self.log_model(model_path, f"trial_{trial_number}_model")
    
    def log_final_model(self, best_params: Dict[str, Any], final_metrics: Dict[str, float], 
                       model_path: str, optimization_results: Dict[str, Any]):
        """Логирует финальную модель с лучшими параметрами"""
        if not self.enabled:
            return
        
        run_name = "final_model"
        tags = {
            "model_type": "lora",
            "stage": "production",
            "optimization_completed": "true"
        }
        
        with self.start_run(run_name=run_name, tags=tags):
            # Логируем лучшие параметры
            self.log_params(best_params)
            
            # Логируем финальные метрики
            self.log_metrics(final_metrics)
            
            # Логируем результаты оптимизации
            self.log_params({
                "n_trials": optimization_results.get("n_trials", 0),
                "best_trial_score": optimization_results.get("best_score", 0.0)
            })
            
            # Логируем финальную модель
            if os.path.exists(model_path):
                self.log_model(model_path, "final_lora_adapter")
            
            # Логируем результаты оптимизации как артефакт
            import json
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(optimization_results, f, indent=2, ensure_ascii=False)
                temp_path = f.name
            
            try:
                self.log_artifact(temp_path, "optimization_results.json")
            finally:
                os.unlink(temp_path)


def create_mlflow_tracker(enabled: bool = None) -> MLflowTracker:
    """Фабрика для создания MLflow трекера"""
    return MLflowTracker(enabled=enabled) 