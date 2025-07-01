# 🚀 Docker Конфигурации FRDA

Данный проект поддерживает два режима развертывания: **CPU-only** и **GPU-accelerated**. Выберите подходящую конфигурацию в зависимости от доступного оборудования.

## 📋 Обзор конфигураций

| Конфигурация | Файл | Описание | Требования |
|--------------|------|----------|------------|
| **CPU** | `docker-compose.cpu.yml` | Все вычисления на CPU, включая обучение LoRA | 8+ GB RAM |
| **GPU** | `docker-compose.gpu.yml` | GPU для обучения, CPU для генерации данных | NVIDIA GPU + Docker GPU поддержка |

## 💻 CPU-Only Конфигурация

### Особенности:
- ✅ **Работает везде** - не требует GPU
- ✅ **Простое развертывание** - стандартный Docker
- ⚠️ **Медленное обучение** - LoRA тренировка на CPU
- ⚠️ **Ограниченные параметры** - меньше эпох и batch size

### Запуск:
```bash
# Запустить все сервисы на CPU
docker-compose -f docker-compose.cpu.yml up -d

# Посмотреть логи
docker-compose -f docker-compose.cpu.yml logs -f

# Остановить
docker-compose -f docker-compose.cpu.yml down
```

### Оптимизированные параметры для CPU:
- `LORA_NUM_EPOCHS: 1` (вместо 3)
- `LORA_BATCH_SIZE: 1` (вместо 8)
- `LORA_N_TRIALS: 3` (вместо 20)
- `OMP_NUM_THREADS: 4` для оптимизации CPU

## 🔥 GPU-Accelerated Конфигурация  

### Особенности:
- 🚀 **Быстрое обучение** - LoRA тренировка на GPU
- 🎯 **Оптимальные параметры** - полные эпохи и большие batch
- 🔬 **Больше экспериментов** - 20 trials для байесовской оптимизации
- ⚡ **Быстрый инференс** - GPU acceleration для плейграунда

### Требования:
1. **NVIDIA GPU** (протестировано на RTX 4090)
2. **NVIDIA Container Toolkit**:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Запуск:
```bash
# Проверить доступность GPU
nvidia-smi

# Запустить все сервисы с GPU
docker-compose -f docker-compose.gpu.yml up -d

# Посмотреть логи GPU воркера
docker-compose -f docker-compose.gpu.yml logs -f gpu-worker

# Остановить
docker-compose -f docker-compose.gpu.yml down
```

### Архитектура GPU конфигурации:
- **CPU Worker** → `cpu_queue` (генерация, валидация данных)
- **GPU Worker** → `gpu_queue` (обучение LoRA адаптеров)
- **GPU Inference** → Быстрый инференс в плейграунде

## 🏗️ Структура файлов

```
├── docker-compose.cpu.yml     # CPU-only конфигурация
├── docker-compose.gpu.yml     # GPU конфигурация
├── worker/
│   ├── Dockerfile.cpu         # CPU воркер
│   ├── Dockerfile.gpu         # GPU воркер
│   └── Dockerfile.uv          # Универсальный (deprecated)
└── inference-worker/
    ├── Dockerfile.cpu         # CPU инференс
    ├── Dockerfile.gpu         # GPU инференс
    └── Dockerfile             # Универсальный (deprecated)
```

## 🔧 Переменные окружения

### Общие для всех конфигураций:
```bash
GOOGLE_API_KEY=your_api_key
GIGACHAT_ACCESS_TOKEN=your_token
```

### GPU-специфичные:
```bash
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0
```

### CPU-специфичные:
```bash
CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=4
```

## 📊 Сравнение производительности

| Операция | CPU | GPU (RTX 4090) | Ускорение |
|----------|-----|----------------|-----------|
| LoRA обучение (1 эпоха) | ~30 мин | ~3 мин | **10x** |
| Байесовская оптимизация | 3 trials | 20 trials | **6.7x** |
| Inference (256 токенов) | ~10 сек | ~1 сек | **10x** |

## 🚀 Быстрый старт

### Для разработки (CPU):
```bash
cp .env.example .env
# Заполните GOOGLE_API_KEY и GIGACHAT_ACCESS_TOKEN
docker-compose -f docker-compose.cpu.yml up -d
```

### Для продакшена (GPU):
```bash
# Проверить GPU
nvidia-smi

# Установить NVIDIA Container Toolkit (если не установлен)
# ... см. раздел требований выше

# Запустить
cp .env.example .env
# Заполните переменные окружения
docker-compose -f docker-compose.gpu.yml up -d
```

## 🐛 Устранение неполадок

### GPU не обнаруживается:
```bash
# Проверить NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Проверить PyTorch в контейнере
docker exec -it data-generation-gpu-worker python -c "import torch; print(torch.cuda.is_available())"
```

### Проблемы с памятью GPU:
```bash
# Уменьшить batch size в .env или docker-compose.gpu.yml
LORA_BATCH_SIZE=4  # вместо 8
LORA_GRAD_ACCUM_STEPS=2  # вместо 1
```

### Медленная работа CPU:
```bash
# Увеличить число потоков
OMP_NUM_THREADS=8  # вместо 4
```

## 📝 Логи и мониторинг

```bash
# Все логи
docker-compose -f docker-compose.gpu.yml logs -f

# Конкретный сервис
docker-compose -f docker-compose.gpu.yml logs -f gpu-worker
docker-compose -f docker-compose.gpu.yml logs -f inference-worker

# Мониторинг GPU
watch -n 1 nvidia-smi
```

---

**Совет:** Для разработки используйте CPU конфигурацию, для продакшена с большими объемами данных - GPU. 