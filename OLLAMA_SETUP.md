# Configuración de Ollama para Survival Analysis

Este documento explica cómo configurar Ollama (Qwen2.5) para que las explicaciones de IA funcionen correctamente.

## ¿Qué es Ollama?

Ollama es un runtime para ejecutar modelos de lenguaje localmente. La aplicación Survival Analysis usa **Qwen2.5** (un modelo de IA chino optimizado) para generar explicaciones automáticas de los análisis de supervivencia.

**Ventajas:**
- No requiere conexión a internet una vez configurado
- Explicaciones privadas (no se envían datos a servidores externos)
- Funciona localmente sin costo

## Requisitos Previos

- **Ollama instalado**: Descargar desde https://ollama.ai
- **Almacenamiento suficiente**: El modelo Qwen2.5 ocupa ~1.7 GB
- **RAM suficiente**: Se recomienda 4GB mínimo

## Paso 1: Iniciar Ollama

**Opción A: Automático (Recomendado)**

Ejecuta el script batch proporcionado:
```
.\START_LLAMA_SERVER.bat
```

Este script:
1. Descarga/prepara el modelo Qwen2.5 automáticamente (si no exista)
2. Inicia el servidor Ollama en http://127.0.0.1:11434
3. Espera a que veas el mensaje "Listening on..." o "API listening on..."

**Opción B: Manual**

Si prefieres controlar el proceso manualmente, abre PowerShell y ejecuta:
```powershell
ollama pull qwen2.5    # Descargar modelo (solo primera vez)
ollama serve           # Iniciar servidor
```

## Paso 2: Verificar que Ollama Funciona

Ejecuta el script de test:
```
python test_ollama_connection.py
```

Este script verifica:
- ✓ Ollama está activo en puerto 11434
- ✓ El modelo Qwen2.5 está disponible
- ✓ El modelo puede generar texto correctamente

**Salida exitosa:**
```
✓ TODOS LOS TESTS PASARON - OLLAMA FUNCIONA CORRECTAMENTE
```

## Paso 3: Ejecutar Survival Analysis

Una vez que Ollama está corriendo:
```
python cargaDataset.py
```

Abre http://localhost:8050 en tu navegador.

## Troubleshooting

### Error: "Connection refused" o "Cannot connect"
**Causa**: El servidor Ollama no está corriendo
**Solución**: 
1. Abre PowerShell
2. Ejecuta: `ollama serve`
3. Espera a ver "API listening on..."
4. Mantén esta ventana abierta mientras usas la app

### Error: "Model not found"
**Causa**: Qwen2.5 no está descargado
**Solución**: Ejecuta `ollama pull qwen2.5` en PowerShell

### Error "timeout" o "too slow"
**Causa**: El modelo está procesando lentamente (normal en la primera pasada)
**Solución**: 
- Intenta con prompts más cortos
- Espera más tiempo (puede tardar 30-60 segundos en la first query)
- Verifica que tienes suficiente RAM disponible

### Puerto 11434 ya está en uso
**Causa**: Otra aplicación usa el mismo puerto
**Solución**:
```powershell
# Ver qué está usando el puerto 11434
netstat -ano | findstr 11434

# Terminar el proceso
taskkill /PID <process_id> /F
```

## Arquitectura de Explicaciones

Cuando haces clic en "Explicar", ocurre esto:

```
[UI: Click "Explicar"] 
    ↓
[cargaDataset.py: responder_pregunta_con_llama3()]
    ↓
[HTTP POST a http://127.0.0.1:11434/api/chat]
    ↓
[Ollama Server: Procesa con modelo Qwen2.5]
    ↓
[Respuesta generada: Explicación del análisis]
    ↓
[Mostrar en UI]
```

## Configuración Actual

- **Endpoint**: http://127.0.0.1:11434/api/chat
- **Modelo**: qwen2.5
- **Puerto**: 11434
- **Timeout**: 10 minutos (600 segundos)

Estos valores están configurados en [cargaDataset.py](cargaDataset.py#L28-L29).

## Referencias

- Sitio oficial de Ollama: https://ollama.ai
- Modelos disponibles: https://ollama.ai/library
- Documentación Qwen: https://github.com/QwenLM/Qwen2.5

---

**Última actualización**: 2024
**Estado**: En producción
