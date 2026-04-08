@echo off
REM Script para iniciar llama-server con Qwen2.5-1.5B-Instruct
REM ===========================================================

setlocal enabledelayedexpansion

REM Rutas al modelo y servidor
set MODEL_PATH=C:\Users\LENOVO\Desktop\IA\qwen2.5-1.5b-instruct-q4_k_m.gguf
set LLAMA_SERVER=C:\Users\LENOVO\Desktop\IA\llama.cpp\llama-server.exe

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
  -ngl 0

pause


  -t 8

REM La aplicación seguirá ejecutándose indefinidamente
REM Presiona Ctrl+C para detener

pause
