@echo off
rem =============================================================================
rem Оптимизированная сборка FRDA с кешированием Docker слоев (Windows)
rem =============================================================================

setlocal

echo 🚀 Запуск оптимизированной сборки FRDA...

rem Включаем экспериментальные функции Docker для лучшего кеширования
set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1

rem =============================================================================
rem ЭТАП 1: Сборка базового образа с зависимостями (кеширование)
rem =============================================================================

echo 📦 Сборка базового образа с Python зависимостями...

rem Сборка только зависимостей для кеширования
docker build ^
    --target dependencies ^
    --tag frda-worker-deps:latest ^
    --file worker/Dockerfile ^
    --build-arg BUILDKIT_INLINE_CACHE=1 ^
    .

if %errorlevel% neq 0 (
    echo ❌ Ошибка сборки базового образа
    exit /b %errorlevel%
)

echo ✅ Базовый образ с зависимостями готов и закеширован

rem =============================================================================
rem ЭТАП 2: Сборка финального образа с кодом
rem =============================================================================

echo 🔧 Сборка финального образа...

rem Используем кеш из предыдущего этапа
docker-compose build ^
    --build-arg BUILDKIT_INLINE_CACHE=1 ^
    cpu-worker

if %errorlevel% neq 0 (
    echo ❌ Ошибка сборки CPU Worker
    exit /b %errorlevel%
)

echo ✅ CPU Worker собран с использованием кеша

rem =============================================================================
rem ЭТАП 3: Опционально - сборка GPU Worker
rem =============================================================================

if "%1"=="--with-gpu" (
    echo 🎮 Сборка GPU Worker...
    docker-compose build ^
        --build-arg BUILDKIT_INLINE_CACHE=1 ^
        gpu-worker
    
    if %errorlevel% neq 0 (
        echo ❌ Ошибка сборки GPU Worker
        exit /b %errorlevel%
    )
    
    echo ✅ GPU Worker собран
)

rem =============================================================================
rem ЭТАП 4: Запуск контейнеров
rem =============================================================================

echo 🚀 Запуск FRDA платформы...

rem Останавливаем существующие контейнеры (если есть)
docker-compose down --remove-orphans

rem Запускаем все сервисы в фоновом режиме
docker-compose up -d

if %errorlevel% neq 0 (
    echo ❌ Ошибка запуска контейнеров
    exit /b %errorlevel%
)

echo ✅ FRDA платформа запущена!

rem Ожидание готовности сервисов
echo ⏳ Ожидание готовности сервисов...
timeout /t 10 /nobreak > nul

rem Проверка статуса
echo 📊 Статус сервисов:
docker-compose ps

rem =============================================================================
rem ИНФОРМАЦИЯ О РЕЗУЛЬТАТЕ
rem =============================================================================

echo.
echo 🎯 Сборка и запуск завершены!
echo.
echo 📊 Информация о кешировании:
echo    • Python зависимости закешированы в образе: frda-worker-deps:latest
echo    • При изменении только кода PyTorch не будет переустанавливаться
echo    • При изменении requirements.txt кеш будет автоматически обновлен
echo.
echo 🌐 Доступные сервисы:
echo    • FastAPI: http://localhost:7777
echo    • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
echo    • PostgreSQL: localhost:5432 (postgres/password)
echo    • Redis: localhost:6379
echo.
echo 🔄 Следующие сборки будут намного быстрее!
echo.
echo 💡 Полезные команды:
echo    docker-compose logs -f        # Логи всех сервисов
echo    docker-compose logs cpu-worker # Логи worker'а
echo    docker-compose down            # Остановить все сервисы
echo    docker images ^| findstr frda   # Посмотреть образы
echo    docker system df               # Посмотреть использование места
echo.

endlocal 