@echo off
echo 🚀 Быстрая сборка с UV + запуск...

set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1

echo 📦 Сборка с UV (в разы быстрее pip)...
docker-compose build cpu-worker

if %errorlevel% neq 0 (
    echo ❌ Ошибка сборки
    exit /b %errorlevel%
)

echo 🚀 Запуск контейнеров...
docker-compose up -d

echo ✅ Готово! Логи:
timeout /t 3 /nobreak > nul
docker-compose logs --tail=10 cpu-worker 