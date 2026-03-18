@echo off
REM Script para iniciar la aplicacion Dash (Survival Analysis)
REM =========================================================

echo.
echo ========================================
echo   INICIANDO SURVIVAL ANALYSIS DASHBOARD
echo ========================================
echo.

REM Cambiar a la carpeta del proyecto
cd /d C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis

REM Verificar que está en la carpeta correcta
if not exist "cargaDataset.py" (
    echo ERROR: No se encontro cargaDataset.py
    echo Por favor verifica la ruta del proyecto
    pause
    exit /b 1
)

echo [✓] Proyecto encontrado
echo.
echo Verificando que llama-server está en ejecución...
echo.

REM Intentar conectar a llama-server
timeout /t 2 /nobreak

echo.
echo ========================================
echo   INICIANDO SERVIDOR DASH
echo ========================================
echo.
echo Accede a: http://localhost:8050
echo.
echo Para detener: Presiona Ctrl+C
echo.

REM Ejecutar la aplicación Dash
python cargaDataset.py

pause
