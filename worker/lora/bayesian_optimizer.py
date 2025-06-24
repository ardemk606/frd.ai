import optuna
from typing import Dict, Any, Callable


class BayesianOptimizer:
    """Класс для байесовской оптимизации гиперпараметров LoRA"""
    
    def __init__(self, n_trials: int = 20):
        """
        Args:
            n_trials: Количество попыток оптимизации
        """
        self.n_trials = n_trials
        self.study = None
    
    def optimize(self, objective_function: Callable[[optuna.Trial], float]) -> Dict[str, Any]:
        """
        Запускает байесовскую оптимизацию
        
        Args:
            objective_function: Функция, которую нужно оптимизировать
            
        Returns:
            Лучшие найденные параметры
        """
        # Создаем study для максимизации метрики
        self.study = optuna.create_study(direction='maximize')
        
        # Запускаем оптимизацию
        self.study.optimize(objective_function, n_trials=self.n_trials)
        
        return self.study.best_params
    
    def suggest_lora_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Предлагает параметры LoRA для оптимизации
        
        Args:
            trial: Объект trial от optuna
            
        Returns:
            Словарь с параметрами LoRA
        """
        rank = trial.suggest_categorical('rank', [8, 16, 32, 64])
        lora_alpha = 2 * rank  # lora_alpha = 2 * rank
        
        return {
            'rank': rank,
            'lora_alpha': lora_alpha,
            'lora_dropout': trial.suggest_float('lora_dropout', 0.0, 0.15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        } 