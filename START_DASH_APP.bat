@echo off
REM Script para iniciar la aplicacion Dash desde la carpeta del proyecto.
REM No usa rutas absolutas del equipo.

setlocal
set "PROJECT_DIR=%~dp0"

echo.
echo ========================================
echo   INICIANDO SURVIVAL ANALYSIS DASHBOARD
echo ========================================
echo.

cd /d "%PROJECT_DIR%"

if not exist "cargaDataset.py" (
    echo ERROR: No se encontro cargaDataset.py en:
    echo %PROJECT_DIR%
    pause
    exit /b 1
)

echo [OK] Proyecto encontrado: %PROJECT_DIR%
echo.
echo ========================================
echo   INICIANDO SERVIDOR DASH
echo ========================================
echo.
echo Accede a: http://127.0.0.1:8050
echo.
echo Para detener: Presiona Ctrl+C
echo.

python -u cargaDataset.py

pause
