"""
Модуль для оценки качества LoRA моделей
"""
import logging
import random
import tempfile
from typing import List, Dict, Any, Tuple
import torch
from bert_score import score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Импорты из нашего проекта
from shared.llm import get_model_by_id

logger = logging.getLogger(__name__)


class BERTScoreEvaluator:
    """Оценка качества через BERTScore"""
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        self.model_name = model_name
        
    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Оценивает качество предсказаний через BERTScore
        
        Args:
            predictions: Сгенерированные тексты
            references: Эталонные тексты
            
        Returns:
            Средний F1 BERTScore (0.0-1.0)
        """
        try:
            # Вычисляем BERTScore
            P, R, F1 = score(
                predictions, 
                references, 
                model_type=self.model_name,
                lang="ru",
                verbose=False
            )
            
            # Возвращаем средний F1
            avg_f1 = F1.mean().item()
            logger.info(f"BERTScore F1: {avg_f1:.4f} (на {len(predictions)} примерах)")
            
            return avg_f1
            
        except Exception as e:
            logger.error(f"Ошибка вычисления BERTScore: {e}")
            return 0.0


class LLMJudgeEvaluator:
    """Оценка качества через LLM Judge"""
    
    def __init__(self, judge_model_id: str = None):
        self.judge_model_id = judge_model_id or "gemini"
        self.evaluation_prompt = self._load_evaluation_prompt()
        
    def _load_evaluation_prompt(self) -> str:
        """Загружает промпт для оценки"""
        try:
            with open("/app/data/prompt/evaluation_prompt.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Не удалось загрузить evaluation prompt: {e}")
            return ""
    
    def evaluate_sample(self, original: str, transformed: str, target_style: str = "Russian street slang") -> float:
        """
        Оценивает один пример через LLM Judge
        
        Args:
            original: Исходный текст
            transformed: Преобразованный текст
            target_style: Описание целевого стиля
            
        Returns:
            Оценка от 0 до 100
        """
        try:
            # Формируем запрос для Judge
            query = f"""**Original Text:** {original}
**Target Style:** {target_style}
**Transformed Text:** {transformed}"""
            
            # Получаем LLM клиент
            llm_client = get_model_by_id(self.judge_model_id)
            if not llm_client:
                logger.error(f"LLM клиент {self.judge_model_id} не найден")
                return 0.0
            
            # Генерируем оценку
            response = llm_client.generate_content(
                system_instruction=self.evaluation_prompt,
                contents=query
            )
            
            # Парсим численную оценку
            try:
                score = float(response.strip())
                return max(0.0, min(100.0, score))  # Ограничиваем 0-100
            except ValueError:
                logger.warning(f"Не удалось распарсить оценку LLM Judge: {response}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Ошибка LLM Judge оценки: {e}")
            return 0.0
    
    def evaluate_batch(self, originals: List[str], transformed: List[str], 
                      target_style: str = "Russian street slang", sample_size: int = 5) -> float:
        """
        Оценивает батч примеров через LLM Judge
        
        Args:
            originals: Исходные тексты
            transformed: Преобразованные тексты
            target_style: Описание целевого стиля
            sample_size: Количество примеров для оценки
            
        Returns:
            Средняя оценка от 0 до 100
        """
        if len(originals) != len(transformed):
            logger.error("Количество исходных и преобразованных текстов не совпадает")
            return 0.0
        
        # Берем случайную выборку для экономии API calls
        sample_size = min(sample_size, len(originals))
        indices = random.sample(range(len(originals)), sample_size)
        
        scores = []
        for i in indices:
            score = self.evaluate_sample(originals[i], transformed[i], target_style)
            scores.append(score)
            logger.debug(f"LLM Judge sample {i}: {score}")
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"LLM Judge средняя оценка: {avg_score:.2f} (на {len(scores)} примерах)")
        
        return avg_score


class ModelEvaluator:
    """Комбинированная оценка модели через BERTScore + LLM Judge"""
    
    def __init__(self, judge_model_id: str = None):
        self.bert_evaluator = BERTScoreEvaluator()
        self.llm_evaluator = LLMJudgeEvaluator(judge_model_id)
        
    def generate_predictions(self, model, tokenizer, validation_data: List[Dict[str, str]], 
                           system_prompt: str = None) -> List[str]:
        """
        Генерирует предсказания модели на validation данных
        
        Args:
            model: Обученная PEFT модель
            tokenizer: Токенизатор
            validation_data: Валидационные данные
            system_prompt: Системный промпт
            
        Returns:
            Список сгенерированных текстов
        """
        predictions = []
        model.eval()
        
        with torch.no_grad():
            for item in validation_data:
                try:
                    # Формируем input prompt
                    system_text = ""
                    if system_prompt and 'system' in item and item['system'] == "$systemPrompt":
                        system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    
                    input_text = f"{system_text}<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n"
                    
                    # Токенизируем
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Генерируем ответ
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Декодируем только новые токены
                    new_tokens = outputs[0][len(inputs["input_ids"][0]):]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    predictions.append(generated_text)
                    
                except Exception as e:
                    logger.warning(f"Ошибка генерации для примера: {e}")
                    predictions.append("")  # Пустой ответ при ошибке
        
        model.train()  # Возвращаем в режим обучения
        return predictions
    
    def quick_evaluate(self, model, tokenizer, validation_data: List[Dict[str, str]], 
                      system_prompt: str = None) -> float:
        """
        Быстрая оценка через BERTScore для байесовской оптимизации
        
        Args:
            model: Обученная PEFT модель
            tokenizer: Токенизатор
            validation_data: Валидационные данные
            system_prompt: Системный промпт
            
        Returns:
            BERTScore F1 (0.0-1.0)
        """
        # Генерируем предсказания
        predictions = self.generate_predictions(model, tokenizer, validation_data, system_prompt)
        
        # Извлекаем эталонные ответы
        references = [item['output'] for item in validation_data]
        
        # Оценка через BERTScore
        return self.bert_evaluator.evaluate(predictions, references)
    
    def detailed_evaluate(self, model, tokenizer, validation_data: List[Dict[str, str]], 
                         system_prompt: str = None, target_style: str = "Russian street slang") -> Dict[str, float]:
        """
        Детальная оценка через BERTScore + LLM Judge
        
        Args:
            model: Обученная PEFT модель
            tokenizer: Токенизатор
            validation_data: Валидационные данные
            system_prompt: Системный промпт
            target_style: Описание целевого стиля
            
        Returns:
            Словарь с метриками
        """
        # Генерируем предсказания
        predictions = self.generate_predictions(model, tokenizer, validation_data, system_prompt)
        
        # Извлекаем данные
        references = [item['output'] for item in validation_data]
        originals = [item['input'] for item in validation_data]
        
        # BERTScore оценка
        bert_score = self.bert_evaluator.evaluate(predictions, references)
        
        # LLM Judge оценка (на выборке)
        llm_score = self.llm_evaluator.evaluate_batch(originals, predictions, target_style)
        
        # Комбинированная метрика
        combined_score = bert_score * 0.4 + (llm_score / 100.0) * 0.6
        
        return {
            'bert_score': bert_score,
            'llm_judge_score': llm_score,
            'combined_score': combined_score
        } 