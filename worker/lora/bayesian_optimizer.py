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
        
        # Получаем лучшие параметры и добавляем lora_alpha
        best_params = self.study.best_params.copy()
        if 'rank' in best_params:
            best_params['lora_alpha'] = 2 * best_params['rank']
        
        return best_params
    
    def suggest_lora_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Предлагает параметры LoRA для оптимизации
        
        Args:
            trial: Объект trial от optuna
            
        Returns:
            Словарь с параметрами LoRA
        """
        rank = trial.suggest_categorical('rank', [8, 16, 32, 64])
        
        # Вычисляем lora_alpha как 2 * rank, но не suggest'им его
        # так как он зависит от rank (будет добавлен вручную)
        lora_alpha = 2 * rank
        
        return {
            'rank': rank,
            'lora_alpha': lora_alpha,
            'lora_dropout': trial.suggest_float('lora_dropout', 0.0, 0.15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        } 