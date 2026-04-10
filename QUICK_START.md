# ARRANQUE RAPIDO - llama-server + Survival Analysis

## Quick Start (4 pasos)

### Paso 1: Inicia llama-server

En una terminal, desde la carpeta del proyecto:

```powershell
cd C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
.\START_LLAMA_SERVER.bat
```

Debes ver que queda escuchando en:

```text
http://127.0.0.1:8000
```

### Paso 2: Verifica el endpoint del modelo

En otra terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    model = "qwen2.5-1.5b-instruct"
    messages = @(@{role = "user"; content = "test"})
    max_tokens = 10
  } | ConvertTo-Json)
```

### Paso 3: Inicia la aplicacion Dash

```powershell
cd C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
python cargaDataset.py
```

O ejecuta:

```text
START_DASH_APP.bat
```

### Paso 4: Abre la app

```text
http://localhost:8050
```

## Scripts disponibles

| Script | Funcion |
|--------|---------|
| START_LLAMA_SERVER.bat | Inicia llama-server con Qwen2.5 |
| START_DASH_APP.bat | Inicia la aplicacion Dash |

## Troubleshooting rapido

### Connection refused

- Verifica que START_LLAMA_SERVER.bat sigue abierto.
- Comprueba puerto: `netstat -ano | findstr :8000`

### Model not found

- Revisa la ruta del modelo en START_LLAMA_SERVER.bat.

### Respuestas lentas

- Ajusta threads con `-t` en START_LLAMA_SERVER.bat (por ejemplo, `-t 8`).

## Nota

Esta version usa llama.cpp/llama-server (endpoint OpenAI-compatible en puerto 8000).
Si necesitas detalle extendido de instalacion, consulta LLAMA_SERVER_SETUP.md.
