@echo off
REM Script para iniciar llama-server con un modelo GGUF local.
REM No usa rutas absolutas del equipo.

setlocal enabledelayedexpansion
set "PROJECT_DIR=%~dp0"
set "DEFAULT_MODEL_PATH=%PROJECT_DIR%models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
set "DEFAULT_LLAMA_SERVER=%PROJECT_DIR%tools\llama.cpp\llama-server.exe"
set "LEGACY_MODEL_PATH=%USERPROFILE%\Desktop\IA\qwen2.5-1.5b-instruct-q4_k_m.gguf"
set "LEGACY_LLAMA_SERVER=%USERPROFILE%\Desktop\IA\llama.cpp\llama-server.exe"

REM Puedes sobrescribir estas rutas antes de ejecutar el script:
REM   set MODEL_PATH=C:\ruta\al\modelo.gguf
REM   set LLAMA_SERVER=C:\ruta\a\llama-server.exe
if "%MODEL_PATH%"=="" (
    if exist "%DEFAULT_MODEL_PATH%" (
        set "MODEL_PATH=%DEFAULT_MODEL_PATH%"
    ) else if exist "%LEGACY_MODEL_PATH%" (
        set "MODEL_PATH=%LEGACY_MODEL_PATH%"
    ) else (
        set "MODEL_PATH=%DEFAULT_MODEL_PATH%"
    )
)

if "%LLAMA_SERVER%"=="" (
    if exist "%DEFAULT_LLAMA_SERVER%" (
        set "LLAMA_SERVER=%DEFAULT_LLAMA_SERVER%"
    ) else if exist "%LEGACY_LLAMA_SERVER%" (
        set "LLAMA_SERVER=%LEGACY_LLAMA_SERVER%"
    ) else (
        set "LLAMA_SERVER=%DEFAULT_LLAMA_SERVER%"
    )
)

REM Parametros de inferencia. Ajusta estos valores segun tu CPU/RAM.
if "%N_THREADS%"=="" set "N_THREADS=8"
if "%N_CTX%"=="" set "N_CTX=4096"
if "%N_BATCH%"=="" set "N_BATCH=512"
if "%N_UBATCH%"=="" set "N_UBATCH=128"

echo.
echo ========================================
echo   INICIANDO LLAMA-SERVER
echo ========================================
echo.

if not exist "%LLAMA_SERVER%" (
    echo ERROR: llama-server.exe no encontrado en:
    echo %LLAMA_SERVER%
    echo.
    echo Ajusta la variable LLAMA_SERVER o coloca llama-server.exe en:
    echo %PROJECT_DIR%tools\llama.cpp\
    pause
    exit /b 1
)

if not exist "%MODEL_PATH%" (
    echo ERROR: Modelo GGUF no encontrado en:
    echo %MODEL_PATH%
    echo.
    echo Ajusta la variable MODEL_PATH o coloca el modelo en:
    echo %PROJECT_DIR%models\
    pause
    exit /b 1
)

echo [OK] Modelo encontrado: %MODEL_PATH%
echo [OK] Servidor encontrado: %LLAMA_SERVER%
echo [i] N_THREADS=%N_THREADS%  N_CTX=%N_CTX%  N_BATCH=%N_BATCH%  N_UBATCH=%N_UBATCH%
echo.
echo Iniciando servidor en: http://127.0.0.1:8000
echo.

"%LLAMA_SERVER%" ^
  -m "%MODEL_PATH%" ^
  --host 127.0.0.1 ^
  --port 8000 ^
  -c %N_CTX% ^
  -b %N_BATCH% ^
  -ub %N_UBATCH% ^
  -ngl 0 ^
  -t %N_THREADS%

pause
