@echo off
setlocal EnableDelayedExpansion
set "EXIT_CODE=0"
pushd "%~dp0" >nul || exit /b 1
REM BGE-M3 Embedding Server - Startup Script
REM Usage: start_server.bat [local^|docker] [cpu^|gpu^|auto]
REM Default: docker with auto-detect (asks if GPU available)

set "MODE=%~1"
set "DEVICE=%~2"
if "%MODE%"=="" set "MODE=docker"
if "%DEVICE%"=="" set "DEVICE=auto"
if "%HOST%"=="" set "HOST=0.0.0.0"

REM Lowercase normalize
for %%A in (LOCAL local Local) do if /I "%MODE%"=="%%A" set "MODE=local"
for %%A in (DOCKER docker Docker) do if /I "%MODE%"=="%%A" set "MODE=docker"
for %%A in (CPU cpu Cpu) do if /I "%DEVICE%"=="%%A" set "DEVICE=cpu"
for %%A in (GPU gpu Gpu) do if /I "%DEVICE%"=="%%A" set "DEVICE=gpu"
for %%A in (AUTO auto Auto) do if /I "%DEVICE%"=="%%A" set "DEVICE=auto"

if not "%MODE%"=="local" if not "%MODE%"=="docker" goto :usage
if not "%DEVICE%"=="cpu" if not "%DEVICE%"=="gpu" if not "%DEVICE%"=="auto" goto :usage

echo ======================================
echo  Select reranker
echo ======================================
echo.
echo Do you want to use:
echo   [1] BGE  ^(BAAI/bge-reranker-v2-m3^)
echo   [2] QWEN ^(Qwen/Qwen3-Reranker-0.6B^)
echo.
set /p reranker_choice="Enter choice ^(1 or 2^): "
if "%reranker_choice%"=="2" (
    set "RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B"
) else (
    if not "%reranker_choice%"=="" if not "%reranker_choice%"=="1" echo [WARNING] Invalid choice, defaulting to BGE
    set "RERANKER_MODEL=BAAI/bge-reranker-v2-m3"
)
echo.

REM Check if CUDA GPU is available
if "%DEVICE%"=="auto" (
    nvidia-smi >nul 2>&1
    if errorlevel 1 goto :no_cuda_detected
    goto :cuda_detected
)
goto :device_selected

:cuda_detected
echo ======================================
echo  CUDA GPU detected on this machine
echo ======================================
echo.
echo Do you want to run with:
echo   [1] GPU ^(faster inference^)
echo   [2] CPU ^(compatible with all systems^)
echo.
set /p choice="Enter choice ^(1 or 2^): "
if "%choice%"=="1" (
    set "DEVICE=gpu"
) else if "%choice%"=="2" (
    set "DEVICE=cpu"
) else (
    echo [WARNING] Invalid choice, defaulting to CPU
    set "DEVICE=cpu"
)
echo.
goto :device_selected

:no_cuda_detected
echo ======================================
echo  WARNING: No CUDA GPU detected
echo ======================================
echo.
echo This machine does not have a CUDA-compatible GPU.
echo The server will run in CPU mode ^(slower but compatible^).
echo.
pause
set "DEVICE=cpu"
echo.
goto :device_selected

:device_selected

echo ========================================
echo  BGE-M3 Embedding Server
echo  Mode:   %MODE%
echo  Device: %DEVICE%
echo  Reranker: %RERANKER_MODEL%
echo ========================================
echo.

if "%MODE%"=="docker" goto :run_docker
goto :run_local

:run_local
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Create it first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    set "EXIT_CODE=1"
    goto :exit
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

python -c "import uvicorn, dotenv" 2>nul
if errorlevel 1 (
    echo [ERROR] uvicorn or python-dotenv not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        set "EXIT_CODE=1"
        goto :exit
    )
)

if "%DEVICE%"=="cpu" (
    echo [INFO] Forcing CPU mode ^(CUDA_VISIBLE_DEVICES=-1^)
    set "CUDA_VISIBLE_DEVICES=-1"
) else (
    echo [INFO] GPU mode ^(CUDA auto-detect^)
    set "CUDA_VISIBLE_DEVICES="
)

set "UVICORN_ENV_ARGS="
if exist ".env" (
    echo [INFO] Loading environment from .env
    set "UVICORN_ENV_ARGS=--env-file .env"
)

echo [INFO] Binding host: %HOST%
echo [INFO] Starting server at http://localhost:8000
echo [INFO] Docs:    http://localhost:8000/docs
echo [INFO] Metrics: http://localhost:8000/metrics
echo Press Ctrl+C to stop
echo.

uvicorn %UVICORN_ENV_ARGS% bge-m3_server:app --host "%HOST%" --port 8000
echo.
echo [INFO] Server stopped
pause
set "EXIT_CODE=0"
goto :exit

:run_docker
where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] docker not found in PATH
    pause
    set "EXIT_CODE=1"
    goto :exit
)

set "COMPOSE_FILES=-f docker-compose.yml"
if "%DEVICE%"=="cpu" (
    if not exist "docker-compose.cpu.yml" (
        echo [ERROR] docker-compose.cpu.yml missing
        pause
        set "EXIT_CODE=1"
        goto :exit
    )
    set "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.cpu.yml"
    echo [INFO] Compose overlay: cpu
) else (
    echo [INFO] Compose overlay: gpu ^(nvidia runtime^)
)

echo [INFO] Building image...
docker compose %COMPOSE_FILES% build
if errorlevel 1 (
    echo [ERROR] Build failed
    pause
    set "EXIT_CODE=1"
    goto :exit
)

echo [INFO] Starting container...
docker compose %COMPOSE_FILES% up -d
if errorlevel 1 (
    echo [ERROR] Container start failed
    pause
    set "EXIT_CODE=1"
    goto :exit
)

echo.
echo [INFO] Container started. Endpoints:
echo   http://localhost:8000/health
echo   http://localhost:8000/docs
echo   http://localhost:8000/metrics
echo.
echo [INFO] Tail logs: docker compose %COMPOSE_FILES% logs -f
echo [INFO] Stop:      docker compose %COMPOSE_FILES% down
echo.
docker compose %COMPOSE_FILES% ps
set "EXIT_CODE=0"
goto :exit

:usage
echo Usage: start_server.bat [local^|docker] [cpu^|gpu^|auto]
echo.
echo Arguments:
echo   local^|docker  - Run mode: local Python or Docker container
echo   cpu^|gpu^|auto  - Device: CPU only, GPU (CUDA), or auto-detect
echo.
echo Examples:
echo   start_server.bat              # docker auto ^(default - detects GPU^)
echo   start_server.bat docker auto  # docker with auto-detect
echo   start_server.bat local cpu
echo   start_server.bat docker gpu
echo   start_server.bat docker cpu
set "EXIT_CODE=1"
goto :exit

:exit
popd >nul
exit /b %EXIT_CODE%
