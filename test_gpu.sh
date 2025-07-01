#!/bin/bash

echo "🧪 Тестирование GPU конфигурации FRDA"
echo "======================================="

# Проверяем nvidia-smi на хосте
echo "1️⃣ Проверка GPU на хосте..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo "✅ GPU обнаружен на хосте"
else
    echo "❌ nvidia-smi не найден. Убедитесь что NVIDIA драйверы установлены."
    exit 1
fi

echo ""

# Проверяем NVIDIA Container Toolkit
echo "2️⃣ Проверка NVIDIA Container Toolkit..."
if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA Container Toolkit работает"
else
    echo "❌ NVIDIA Container Toolkit не настроен. Установите его:"
    echo "   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
    echo "   distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "   curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
    echo "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "   sudo systemctl restart docker"
    exit 1
fi

echo ""

# Запускаем GPU конфигурацию для тестирования
echo "3️⃣ Запуск GPU воркера для тестирования..."
echo "   (Это может занять несколько минут при первом запуске)"

# Запускаем только GPU воркер для быстрого теста
docker-compose -f docker-compose.gpu.yml up -d postgres redis minio minio-setup

# Ждем пока инфраструктура запустится
echo "   Ожидание запуска инфраструктуры..."
sleep 10

# Запускаем GPU воркер
docker-compose -f docker-compose.gpu.yml up -d gpu-worker

echo ""
echo "4️⃣ Проверка PyTorch в GPU воркере..."
sleep 20  # Даем время воркеру запуститься

if docker exec data-generation-gpu-worker python -c "import torch; print('✅ PyTorch доступен:', torch.__version__); print('✅ CUDA доступна:', torch.cuda.is_available()); print('✅ GPU устройств:', torch.cuda.device_count()); print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null; then
    echo "✅ GPU воркер настроен правильно!"
else
    echo "❌ Проблемы с PyTorch в GPU воркере"
    echo "   Логи воркера:"
    docker-compose -f docker-compose.gpu.yml logs gpu-worker | tail -20
fi

echo ""

# Проверяем inference worker
echo "5️⃣ Проверка inference worker..."
docker-compose -f docker-compose.gpu.yml up -d inference-worker
sleep 10

if docker exec inference-worker-gpu python -c "import torch; print('✅ Inference PyTorch:', torch.__version__); print('✅ Inference CUDA:', torch.cuda.is_available())" 2>/dev/null; then
    echo "✅ Inference worker настроен правильно!"
else
    echo "❌ Проблемы с inference worker"
    echo "   Логи inference worker:"
    docker-compose -f docker-compose.gpu.yml logs inference-worker | tail -10
fi

echo ""
echo "🎉 Тестирование завершено!"
echo ""
echo "📊 Для мониторинга GPU используйте:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "📝 Для просмотра логов:"
echo "   docker-compose -f docker-compose.gpu.yml logs -f gpu-worker"
echo "   docker-compose -f docker-compose.gpu.yml logs -f inference-worker"
echo ""
echo "🛑 Для остановки:"
echo "   docker-compose -f docker-compose.gpu.yml down" 