@echo off
REM Script para iniciar llama-server con Qwen2.5-1.5B-Instruct
REM ===========================================================

setlocal enabledelayedexpansion

REM Rutas al modelo y servidor
if "%MODEL_PATH%"=="" set MODEL_PATH=C:\Users\LENOVO\Desktop\IA\qwen2.5-1.5b-instruct-q4_k_m.gguf
if "%LLAMA_SERVER%"=="" set LLAMA_SERVER=C:\Users\LENOVO\Desktop\IA\llama.cpp\llama-server.exe

REM ==========================
REM Parametros de inferencia
REM ==========================
REM Ajusta estos valores segun tu CPU/RAM.
if "%N_THREADS%"=="" set N_THREADS=8
if "%N_CTX%"=="" set N_CTX=4096
if "%N_BATCH%"=="" set N_BATCH=512
if "%N_UBATCH%"=="" set N_UBATCH=128

echo.
echo ========================================
echo   INICIANDO LLAMA-SERVER
echo ========================================
echo.

REM Verificar que llama-server existe
if not exist "%LLAMA_SERVER%" (
    echo ERROR: llama-server.exe no encontrado en:
    echo %LLAMA_SERVER%
    echo.
    echo Descargar desde: https://github.com/ggerganov/llama.cpp/releases
    pause
    exit /b 1
)

REM Verificar que el modelo existe
if not exist "%MODEL_PATH%" (
    echo ERROR: Modelo no encontrado en:
    echo %MODEL_PATH%
    pause
    exit /b 1
)

echo [✓] Modelo encontrado: %MODEL_PATH%
echo [✓] Servidor encontrado: %LLAMA_SERVER%
echo [i] N_THREADS=%N_THREADS%  N_CTX=%N_CTX%  N_BATCH=%N_BATCH%  N_UBATCH=%N_UBATCH%
echo.
echo Iniciando servidor en: http://127.0.0.1:8000
echo.
echo ========================================
echo ESPERANDO SERVIDOR...
echo ========================================
echo.

REM Ejecutar llama-server (sin límite de tiempo)
"%LLAMA_SERVER%" ^
  -m "%MODEL_PATH%" ^
  --host 127.0.0.1 ^
  --port 8000 ^
  -c %N_CTX% ^
  -b %N_BATCH% ^
  -ub %N_UBATCH% ^
  -ngl 0 ^
  -t %N_THREADS%

REM La aplicación seguirá ejecutándose indefinidamente
REM Presiona Ctrl+C para detener

pause
