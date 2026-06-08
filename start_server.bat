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
REM PORT: shell env wins; else read from .env; else default 8000.
if "%PORT%"=="" if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%K in (".env") do (
        if /I "%%K"=="PORT" set "PORT=%%L"
    )
)
if "%PORT%"=="" set "PORT=8000"

REM Lowercase normalize
for %%A in (LOCAL local Local) do if /I "%MODE%"=="%%A" set "MODE=local"
for %%A in (DOCKER docker Docker) do if /I "%MODE%"=="%%A" set "MODE=docker"
for %%A in (CPU cpu Cpu) do if /I "%DEVICE%"=="%%A" set "DEVICE=cpu"
for %%A in (GPU gpu Gpu) do if /I "%DEVICE%"=="%%A" set "DEVICE=gpu"
for %%A in (AUTO auto Auto) do if /I "%DEVICE%"=="%%A" set "DEVICE=auto"

if not "%MODE%"=="local" if not "%MODE%"=="docker" goto :usage
if not "%DEVICE%"=="cpu" if not "%DEVICE%"=="gpu" if not "%DEVICE%"=="auto" goto :usage

echo ======================================
echo  Select dense embedding backend
echo ======================================
echo.
echo Do you want dense embeddings to use:
echo   [1] BGE  ^(BAAI/bge-m3^)
echo   [2] QWEN ^(Qwen/Qwen3-Embedding-0.6B^)
echo.
set /p dense_choice="Enter choice (1 or 2): "
if "%dense_choice%"=="2" (
    set "DENSE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B"
) else (
    if not "%dense_choice%"=="" if not "%dense_choice%"=="1" echo [WARNING] Invalid choice, defaulting dense embeddings to BGE
    set "DENSE_EMBEDDING_MODEL=BAAI/bge-m3"
)
echo.

echo ======================================
echo  Select reranker
echo ======================================
echo.
echo Do you want to use:
echo   [1] BGE  ^(BAAI/bge-reranker-v2-m3^)
echo   [2] QWEN ^(Qwen/Qwen3-Reranker-0.6B^)
echo.
set /p reranker_choice="Enter choice (1 or 2): "
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
set /p choice="Enter choice (1 or 2): "
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

if "%DEVICE%"=="gpu" if "%RERANKER_MODEL%"=="Qwen/Qwen3-Reranker-0.6B" call :autotune_qwen_reranker_gpu
if "%DEVICE%"=="cpu" if "%RERANKER_MODEL%"=="Qwen/Qwen3-Reranker-0.6B" call :autotune_qwen_reranker_cpu

echo ========================================
echo  BGE-M3 Embedding Server
echo  Mode:   %MODE%
echo  Device: %DEVICE%
echo  Dense:  %DENSE_EMBEDDING_MODEL%
echo  Reranker: %RERANKER_MODEL%
if "%RERANKER_MODEL%"=="Qwen/Qwen3-Reranker-0.6B" (
    echo  Qwen rerank batch: %QWEN_RERANK_BATCH_SIZE%
    echo  Qwen rerank max length: %QWEN_RERANK_MAX_LENGTH%
)
echo ========================================
echo.

if "%MODE%"=="docker" goto :run_docker
goto :run_local

:run_local
REM Pick CPU-only requirements when running without GPU to avoid the CUDA torch wheel.
set "REQ_FILE=requirements-gpu.txt"
if "%DEVICE%"=="cpu" set "REQ_FILE=requirements-cpu.txt"
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Create it first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r %REQ_FILE%
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
    pip install -r %REQ_FILE%
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
echo [INFO] Starting server at http://localhost:%PORT%
echo [INFO] Docs:    http://localhost:%PORT%/docs
echo [INFO] Metrics: http://localhost:%PORT%/metrics
echo Press Ctrl+C to stop
echo.

uvicorn %UVICORN_ENV_ARGS% bge-m3_server:app --host "%HOST%" --port %PORT%
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

set "COMPOSE_FILES=-f docker-compose.gpu.yml"
if "%DEVICE%"=="cpu" (
    if not exist "docker-compose.cpu.yml" (
        echo [ERROR] docker-compose.cpu.yml missing
        pause
        set "EXIT_CODE=1"
        goto :exit
    )
    set "COMPOSE_FILES=-f docker-compose.gpu.yml -f docker-compose.cpu.yml"
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
echo   http://localhost:%PORT%/health
echo   http://localhost:%PORT%/docs
echo   http://localhost:%PORT%/metrics
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

:autotune_qwen_reranker_gpu
call :load_env_defaults_for_autotune

set "GPU_MEM_MIB="
for /f "usebackq tokens=* delims=" %%M in (`nvidia-smi --query-gpu^=memory.total --format^=csv,noheader,nounits 2^>nul`) do (
    if not defined GPU_MEM_MIB set "GPU_MEM_MIB=%%M"
)
set "GPU_MEM_MIB=!GPU_MEM_MIB: =!"

if "!GPU_MEM_MIB!"=="" (
    echo [WARNING] Could not detect GPU VRAM; keeping Qwen rerank defaults.
    if "!QWEN_RERANK_BATCH_SIZE!"=="" set "QWEN_RERANK_BATCH_SIZE=16"
    if "!QWEN_RERANK_MAX_LENGTH!"=="" set "QWEN_RERANK_MAX_LENGTH=8192"
    goto :eof
)

set "TUNED_QWEN_RERANK_BATCH_SIZE=16"
set "TUNED_QWEN_RERANK_MAX_LENGTH=8192"
if !GPU_MEM_MIB! LEQ 6144 (
    set "TUNED_QWEN_RERANK_BATCH_SIZE=4"
    set "TUNED_QWEN_RERANK_MAX_LENGTH=4096"
) else if !GPU_MEM_MIB! LEQ 8192 (
    set "TUNED_QWEN_RERANK_BATCH_SIZE=8"
    set "TUNED_QWEN_RERANK_MAX_LENGTH=8192"
)

if "!QWEN_RERANK_BATCH_SIZE!"=="" (
    set "QWEN_RERANK_BATCH_SIZE=!TUNED_QWEN_RERANK_BATCH_SIZE!"
    echo [INFO] Auto-tuned QWEN_RERANK_BATCH_SIZE=!QWEN_RERANK_BATCH_SIZE! for !GPU_MEM_MIB!MiB VRAM
) else (
    echo [INFO] Keeping QWEN_RERANK_BATCH_SIZE=!QWEN_RERANK_BATCH_SIZE! ^(user/env override^)
)

if "!QWEN_RERANK_MAX_LENGTH!"=="" (
    set "QWEN_RERANK_MAX_LENGTH=!TUNED_QWEN_RERANK_MAX_LENGTH!"
    echo [INFO] Auto-tuned QWEN_RERANK_MAX_LENGTH=!QWEN_RERANK_MAX_LENGTH! for !GPU_MEM_MIB!MiB VRAM
) else (
    echo [INFO] Keeping QWEN_RERANK_MAX_LENGTH=!QWEN_RERANK_MAX_LENGTH! ^(user/env override^)
)
goto :eof

:autotune_qwen_reranker_cpu
call :load_env_defaults_for_autotune

if "!QWEN_RERANK_BATCH_SIZE!"=="" (
    set "QWEN_RERANK_BATCH_SIZE=1"
    echo [INFO] Auto-tuned QWEN_RERANK_BATCH_SIZE=!QWEN_RERANK_BATCH_SIZE! for CPU mode
) else (
    echo [INFO] Keeping QWEN_RERANK_BATCH_SIZE=!QWEN_RERANK_BATCH_SIZE! ^(user/env override^)
)

if "!QWEN_RERANK_MAX_LENGTH!"=="" (
    set "QWEN_RERANK_MAX_LENGTH=2048"
    echo [INFO] Auto-tuned QWEN_RERANK_MAX_LENGTH=!QWEN_RERANK_MAX_LENGTH! for CPU mode
) else (
    echo [INFO] Keeping QWEN_RERANK_MAX_LENGTH=!QWEN_RERANK_MAX_LENGTH! ^(user/env override^)
)
goto :eof

:load_env_defaults_for_autotune
if not exist ".env" goto :eof

for /f "usebackq tokens=1,* delims==" %%K in (".env") do (
    if /I "%%K"=="QWEN_RERANK_BATCH_SIZE" if "!QWEN_RERANK_BATCH_SIZE!"=="" set "QWEN_RERANK_BATCH_SIZE=%%L"
    if /I "%%K"=="QWEN_RERANK_MAX_LENGTH" if "!QWEN_RERANK_MAX_LENGTH!"=="" set "QWEN_RERANK_MAX_LENGTH=%%L"
)
goto :eof

:exit
popd >nul
exit /b %EXIT_CODE%
