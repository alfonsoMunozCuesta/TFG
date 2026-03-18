# Integración Llama.cpp (llama-server) - Qwen2.5-1.5B-Instruct

## 📋 Descripción General

Se integró un **LLM local de baja latencia** basado en **Qwen2.5-1.5B-Instruct** cuantizado en **GGUF (Q4_K_M)** ejecutado mediante **llama.cpp** (llama-server) y consumido desde la aplicación Dash vía endpoint HTTP local.

**Ventajas:**
- ✅ Sin dependencias de APIs externas
- ✅ Rendimiento optimizado en CPU
- ✅ Baja latencia en respuestas
- ✅ Privacidad garantizada (todo local)
- ✅ Compatible con Windows, macOS y Linux

---

## 🚀 Instalación y Configuración

### 1. Descargar Llama.cpp

Desde GitHub: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

**Opción A: Compilado (Recomendado para Windows)**
```bash
# Descarga pre-compilado desde:
# https://github.com/ggerganov/llama.cpp/releases
# Busca: llama-b<version>-bin-win-avx2.zip (o tu arquitectura)
```

**Opción B: Python Package**
```bash
pip install llama-cpp-python
pip install llama-cpp-python[server]
```

### 2. Descargar Modelo Qwen2.5-1.5B-Instruct (GGUF)

El modelo debe estar cuantizado en formato GGUF (Q4_K_M):

```bash
# Opción 1: Desde HuggingFace (Recomendado)
# Descarga from: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF

# O usa: pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models

# Opción 2: Descargar manualmente
# https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 3. Estructura de Directorios

```
Survival-Analysis/
├── models/
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf    # Modelo GGUF
├── cargaDataset.py
├── ollama_AI.py
└── LLAMA_SERVER_SETUP.md                     # Este archivo
```

---

## 🔧 Ejecutar Llama-Server

### Opción 1: Usando llama-cpp-python (Recomendado)

```bash
# 1. Instalar el paquete con soporte server
pip install llama-cpp-python[server]

# 2. Ejecutar el servidor
python -m llama_cpp.server \
  --model ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8000 \
  --n_gpu_layers 0 \
  --n_threads 4 \
  --chat_format chatml
```

### Opción 2: Usando Binario Compilado de Llama.cpp

```bash
# Desde el directorio de llama.cpp
./llama-server.exe \
  -m ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8000 \
  -ngl 0 \
  -t 4 \
  -ubatch 512
```

### Parámetros Importantes

| Parámetro | Descripción |
|-----------|------------|
| `--model` | Ruta al archivo GGUF del modelo |
| `--host` | Dirección del servidor (127.0.0.1 = local) |
| `--port` | Puerto HTTP (default: 8000) |
| `-ngl` / `--n_gpu_layers` | Capas GPU (0 = solo CPU) |
| `-t` / `--n_threads` | Threads CPU (ajusta según tu CPU) |
| `--chat_format` | Formato de chat (chatml para Qwen2.5) |

---

## ✅ Verificar Servidor

Una vez ejecutado, verifica que funciona:

```bash
# 1. Browser: Abre en el navegador
http://127.0.0.1:8000/docs

# 2. Terminal: Envía una solicitud de prueba
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}],
    "temperature": 0.2,
    "max_tokens": 100
  }'

# Respuesta esperada:
# {"id": "chatcmpl-...", "choices": [{"message": {"content": "..."}}]}
```

---

## 🎯 Integración con la Aplicación Dash

### Archivos Modificados

#### 1. **ollama_AI.py** (Reescrito)
```python
import requests

LLAMA_SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

def generate_explanation(graph_data, model_type):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 3000,
        "stream": False
    }
    response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=30)
    # ...
```

#### 2. **cargaDataset.py** (Actualizado)
- Eliminada importación: `from ollama import Client`
- Agregada importación: `import requests`
- Función `responder_pregunta_con_llama3()` ahora usa endpoint HTTP

### Configuración

```python
# En cargaDataset.py
LLAMA_SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"
```

Si necesitas cambiar host o puerto, edita estas variables.

---

## 🚀 Ejecutar el Proyecto

### 1. Iniciar Llama-Server (en una terminal)
```bash
python -m llama_cpp.server \
  --model ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8000 \
  --n_threads 4
```

### 2. Iniciar Aplicación Dash (en otra terminal)
```bash
cd c:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis
python cargaDataset.py
```

### 3. Acceder a la Aplicación
```
http://localhost:8050
```

---

## 📊 Endpoint OpenAI-Compatible

Llama-server implementa la API compatible con OpenAI. Puedes usarla directamente:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "qwen2.5-1.5b-instruct",
        "messages": [
            {"role": "system", "content": "Eres un experto en análisis de supervivencia."},
            {"role": "user", "content": "Explica Kaplan-Meier"}
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
        "top_p": 0.95,
        "stream": False
    }
)

print(response.json()['choices'][0]['message']['content'])
```

---

## 🔍 Troubleshooting

### Error: "Connection refused"
```
Solución: Asegúrate de que llama-server está ejecutándose
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows
```

### Error: "Model not found"
```
Solución: Verifica que el archivo GGUF existe en la ruta correcta
Edita la ruta en ollama_AI.py y cargaDataset.py si es necesario
```

### Respuestas Lentas
```
Soluciones:
1. Aumenta --n_threads (ej: 8 threads para CPU de 8 núcleos)
2. Reduce max_tokens en los prompts
3. Verifica CPU/RAM disponible durante ejecución
```

### Salida en Otros Idiomas
```
El modelo Qwen2.5 responde en el idioma del prompt
Para garantizar español, comienza prompts con: "Por favor, en español:"
```

---

## 📈 Rendimiento Esperado

Con **Qwen2.5-1.5B-Instruct Q4_K_M**:
- **Tamaño del Modelo:** ~1.1 GB RAM
- **Latencia Primera Respuesta:** 2-5 segundos (según CPU)
- **Velocidad de Generación:** 10-30 tokens/segundo (CPU single-thread)
- **Recomendado:** CPU con 4+ núcleos

---

## 📚 Referencias

- [Llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Qwen2.5 HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [OpenAI API Compatible](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)

---

## ✨ Cambios desde Ollama/Llama3

| Aspecto | Antes (Ollama) | Ahora (Llama.cpp) |
|--------|---------------|-----------------|
| **Herramienta** | Ollama | Llama.cpp (llama-server) |
| **Modelo** | Llama3 | Qwen2.5-1.5B-Instruct |
| **Puerto** | 11434 | 8000 |
| **Cliente** | `ollama.Client()` | `requests.post()` |
| **Dependencia** | ollama Python package | requests (ya incluido) |
| **Consumo RAM** | 7-8 GB | ~1.1 GB |
| **Velocidad** | Más lento | Más rápido (optimizado) |

---

**Última actualización:** 7 de Marzo, 2026
