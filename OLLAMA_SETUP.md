# Nota de compatibilidad: OLLAMA_SETUP.md

Este archivo se mantiene por compatibilidad de nombre, pero el proyecto actual ya no usa Ollama como backend principal.

## Stack actual

- Runtime LLM: llama.cpp (llama-server)
- Endpoint: http://127.0.0.1:8000/v1/chat/completions
- Modelo: qwen2.5-1.5b-instruct

## Que usar a partir de ahora

1. Para instalacion y configuracion completa: LLAMA_SERVER_SETUP.md
2. Para arranque rapido: QUICK_START.md
3. Para ejecutar: START_LLAMA_SERVER.bat + START_DASH_APP.bat

## Motivo de este cambio

La documentacion previa de Ollama apuntaba al puerto 11434 y al endpoint /api/chat, lo cual no coincide con la implementacion actual de la app.
La aplicacion esta conectada a llama-server en puerto 8000 con API OpenAI-compatible.
