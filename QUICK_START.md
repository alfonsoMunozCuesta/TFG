# 🚀 ARRANQUE RÁPIDO - Ollama + Survival Analysis

## ⚡ Quick Start (4 pasos)

### Paso 0: Verificar Prerequisites

1. **Ollama instalado**: Descargar desde https://ollama.ai
2. **Qwen2.5 disponible**: El script descargará automáticamente si falta

---

### Paso 1: Abre Terminal 1 - Inicia Ollama Server

**Opción A (Recomendado - Automático):**
```powershell
cd C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
.\START_LLAMA_SERVER.bat
```

**Opción B (Manual - PowerShell):**
```powershell
ollama pull qwen2.5    # Descargar modelo (solo primera vez)
ollama serve           # Iniciar servidor
```

**Espera este mensaje:**
```
Listening on 127.0.0.1:11434
```

---

### Paso 2: Abre Terminal 2 - Verifica Ollama

En otra ventana de terminal, ejecuta el test de conexión:
```powershell
cd C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
python test_ollama_connection.py
```

Deberías ver:
```
✓ TODOS LOS TESTS PASARON - OLLAMA FUNCIONA CORRECTAMENTE
```

Si algo falla, consulta [OLLAMA_SETUP.md](OLLAMA_SETUP.md#troubleshooting).

---

### Paso 3: Abre Terminal 3 - Inicia Dash App

```powershell
cd C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
python cargaDataset.py
```

**O doble-click:**
```
START_DASH_APP.bat
```

---

### Paso 4: Accede a la App

```
🌐 http://localhost:8050
```

---

## 📝 Notas

- **Ollama debe estar corriendo todo el tiempo** que uses la app
- **Las explicaciones de IA requieren que Ollama esté activo**
- Para más detalles, ver [OLLAMA_SETUP.md](OLLAMA_SETUP.md)

✅ **¡Listo!** Comenzarás a ver preguntas al LLM automáticamente.

---

## 📂 Ubicaciones de Archivos

```
✓ Modelo GGUF:     C:\Users\LENOVO\Desktop\IA\qwen2.5-1.5b-instruct-q4_k_m.gguf
✓ Llama-server:    C:\Users\LENOVO\Desktop\IA\llama.cpp\llama-server.exe
✓ Proyecto:        C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
```

---

## 🔧 Scripts de Ayuda

| Script | Función |
|--------|---------|
| `setup_llama_server.py` | Verifica archivos y inicia servidor |
| `START_LLAMA_SERVER.bat` | Inicia servidor (modo manual) |
| `START_DASH_APP.bat` | Inicia aplicación Dash |

---

## ⚠️ Troubleshooting

### ❌ "Connection refused"
```
→ Asegúrate de que Terminal 1 (llama-server) está ejecutándose
→ Espera 10-15 segundos después de iniciar
→ Verifica que Puerto 8000 está libre: netstat -ano | findstr :8000
```

### ❌ "Model not found"
```
→ Verifica que el archivo GGUF existe en:
   C:\Users\LENOVO\Desktop\IA\qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### ❌ "App slow / freezing"
```
→ Aumenta threads: cambiar "-t 4" a "-t 8" en los comandos
→ Verifica RAM disponible (necesita ~2-3 GB)
```

### ❌ "Respuestas en inglés"
```
→ El modelo responde en el idioma del prompt
→ Los prompts en la app comienzan con "Por favor, en español:"
```

---

## 🔍 Verificar Estado

**En navegador:**
```
http://127.0.0.1:8000/docs
```

**En PowerShell:**
```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    model = "qwen2.5-1.5b-instruct"
    messages = @(@{role = "user"; content = "test"})
    max_tokens = 10
  } | ConvertTo-Json)

$response
```

---

## 📊 Configuración de Recursos

**Threads CPU (cambiar `-t` en comandos):**
```
-t 2   = CPU bajo (2 núcleos)
-t 4   = CPU estándar (4 núcleos) ← DEFAULT
-t 8   = CPU alto (8+ núcleos)
-t 16  = CPU muy alto (16+ núcleos)
```

**Por GPU (no disponible en Win 10 standard):**
```
-ngl 0  = Solo CPU (actual)
-ngl 33 = Transferir todas las capas a GPU (si disponible)
```

---

## 📝 Logs y Diagnóstico

**Para ver más detalles:**

1. Abre `setup_llama_server.py` en VSCode
2. Descomenta línea 145: `print(f"Debug: {result}")` 
3. Vuelve a ejecutar

---

## ✨ Características

- ✅ **Baja latencia:** Respuestas en 2-5 segundos
- ✅ **Privacidad:** Todo local, sin APIs externas
- ✅ **Rendimiento:** Optimizado para CPU
- ✅ **Multilingüe:** Español e Inglés automáticos
- ✅ **Análisis automático:** LLM genera explicaciones de gráficos

---

**Versión:** 1.0 | Fecha: 7 de Marzo, 2026
**Modelo:** Qwen2.5-1.5B-Instruct (GGUF Q4_K_M)
**Framework:** Llama.cpp (llama-server)
