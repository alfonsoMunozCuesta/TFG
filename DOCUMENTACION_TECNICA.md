# Documentación Técnica — Survival Analysis Dashboard

---

## 1. Soporte Multilingüe (Español / Inglés)

### 1.1 Arquitectura

El sistema multilingüe se basa en **3 componentes**:

| Componente | Archivo | Función |
|---|---|---|
| Diccionario de traducciones | `translations.py` | Almacena todos los textos en ES y EN |
| Store de idioma | `cargaDataset.py` | Guarda el idioma seleccionado en memoria |
| Páginas dinámicas | `layout.py` | Genera las páginas con textos del idioma activo |

### 1.2 Diccionario de traducciones (`translations.py`)

Estructura de 2 niveles: **idioma → clave → texto**

```python
translations = {
    'es': {
        'kaplan_meier': 'Kaplan-Meier',
        'genero': 'Género',
        'explicar_kaplan': 'Explicar',
        # ...más claves
    },
    'en': {
        'kaplan_meier': 'Kaplan-Meier',
        'genero': 'Gender',
        'explicar_kaplan': 'Explain',
        # ...más claves
    }
}

def get_translation(lang, key):
    """Busca la clave en el idioma indicado. Si no existe, devuelve la clave."""
    return translations.get(lang, {}).get(key, key)
```

### 1.3 Store de idioma (`cargaDataset.py`)

Un `dcc.Store` es un componente invisible de Dash que guarda datos en memoria del navegador:

```python
dcc.Store(id='language-store', data='es')  # Por defecto español
```

Se actualiza con un callback cuando el usuario pulsa ES o EN:

```python
@app.callback(Output('language-store', 'data'), Input('language-selector', 'value'))
def update_language(selected_language):
    return selected_language
```

### 1.4 Páginas dinámicas (`layout.py`)

Cada página es una **función** que recibe el idioma y usa `get_translation()`:

```python
def create_kaplan_meier_page(language='es'):
    return html.Div([
        html.H1(get_translation(language, 'kaplan_meier')),
        html.Button(get_translation(language, 'explicar_kaplan'), ...),
    ])
```

### 1.5 Flujo completo

```
 Usuario pulsa "EN"
     │
     ▼
 RadioItems value='en'
     │
     ▼
 Callback update_language()  →  language-store = 'en'
     │
     ▼
 Callback display_page() se RE-EJECUTA (language-store es un Input)
     │
     ▼
 Llama a create_kaplan_meier_page('en')
     │
     ▼
 Cada texto usa get_translation('en', clave)
     │
     ▼
 Busca en translations['en'][clave]  →  Devuelve texto en inglés
     │
     ▼
 Página renderizada en INGLÉS
```

**Clave**: `language-store` es un `Input` del callback `display_page`. Cuando cambia, Dash automáticamente re-ejecuta el callback y reconstruye la página.

### 1.6 Cómo añadir un nuevo texto traducible

1. Añadir la clave en `translations.py` en ambos idiomas:
   ```python
   'es': { 'mi_nuevo_texto': 'Hola mundo' },
   'en': { 'mi_nuevo_texto': 'Hello world' }
   ```
2. Usarla en el layout:
   ```python
   html.H2(get_translation(language, 'mi_nuevo_texto'))
   ```

---

## 2. Integración de IA (Qwen2.5 + llama.cpp)

### 2.1 ¿Qué modelo se usa?

**Qwen2.5-1.5B-Instruct** — modelo de lenguaje de 1.5 mil millones de parámetros, cuantizado en formato GGUF (Q4_K_M) para funcionar en CPU sin GPU dedicada.

### 2.2 ¿Qué herramienta lo ejecuta?

**llama.cpp** — proyecto open-source en C++ que ejecuta modelos de lenguaje localmente. Usamos `llama-server.exe`, que levanta un servidor HTTP local compatible con la API de OpenAI.

### 2.3 Descarga e instalación

#### Paso 1: Descargar llama.cpp
```
Ir a: https://github.com/ggerganov/llama.cpp/releases
Descargar: llama-<version>-bin-win-cpu-x64.zip
Descomprimir en: C:\Users\LENOVO\Desktop\IA\llama.cpp\
```
El archivo clave es `llama-server.exe`.

#### Paso 2: Descargar el modelo Qwen2.5
```
Ir a: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
Descargar: qwen2.5-1.5b-instruct-q4_k_m.gguf
Guardar en: C:\Users\LENOVO\Desktop\IA\
```

### 2.4 Organización de archivos

```
Survival-Analysis/
├── config.py               ← URL del servidor y nombre del modelo
├── ollama_AI.py            ← Módulo que envía peticiones al servidor
├── cargaDataset.py         ← Callbacks que preparan datos y llaman a la IA
├── START_LLAMA_SERVER.bat  ← Script para arrancar el servidor
└── START_DASH_APP.bat      ← Script para arrancar el dashboard
```

### 2.5 Configuración (`config.py`)

```python
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"
```

### 2.6 Arranque del servidor (`START_LLAMA_SERVER.bat`)

```bash
llama-server.exe
  -m "qwen2.5-1.5b-instruct-q4_k_m.gguf"   # Ruta al modelo
  --host 127.0.0.1                           # Solo acceso local
  --port 8080                                # Puerto donde escucha
  -ngl 0                                     # 0 capas en GPU (todo CPU)
  -t 8                                       # 8 threads de CPU
```

### 2.7 Módulo de comunicación con la IA (`ollama_AI.py`)

Contiene una única función `generate_explanation(prompt, language)`:

1. **System prompt**: Define el rol de la IA como analista de supervivencia
2. **Idioma**: Ajusta la instrucción según ES/EN
3. **Petición HTTP**: Envía POST a `127.0.0.1:8080/v1/chat/completions` con formato OpenAI
4. **Parámetros**: `temperature=0.2` (preciso), `max_tokens=1200`

```python
payload = {
    "model": "qwen2.5-1.5b-instruct",
    "messages": [
        {"role": "system", "content": "Eres un analista de datos..."},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.2,
    "max_tokens": 1200,
}
response = requests.post(LLAMA_SERVER_URL, json=payload)
```

### 2.8 Cómo se preparan los datos para la IA (`cargaDataset.py`)

Los callbacks **NO** envían el JSON crudo de los gráficos. En su lugar, **calculan estadísticas limpias** directamente del dataset usando `lifelines`:

#### Kaplan-Meier
```
Se detecta qué botón pulsó el usuario (Género/Discapacidad/Ninguna)
→ Se calculan: n, mediana de supervivencia, S(50), S(100) por grupo
→ Prompt: "Análisis KM agrupado por Género. Masculino: n=1265, median=inf..."
```

#### Regresión de Cox
```
Se lee el dropdown (qué covariables seleccionó el usuario)
→ Se ejecuta CoxPHFitter con esas covariables
→ Se extraen: coef, HR (Hazard Ratio), p-value
→ Prompt: "Resultados Cox. Género: coef=0.12, HR=1.13, p=0.04..."
```

#### Test de Log-Rank
```
Se lee el dropdown (qué covariables seleccionó)
→ Se ejecuta el test para cada covariable
→ Se extraen: test_statistic, p_value, decisión
→ Prompt: "Resultados Log-Rank. Género: stat=28.5, p=0.000..."
```

### 2.9 Flujo completo cuando el usuario pulsa "Explicar"

```
 Usuario pulsa "Explicar" en Kaplan-Meier
     │
     ▼
 CALLBACK explicar_kaplan() en cargaDataset.py
     │
     ├── 1) Detecta qué covariable está seleccionada
     ├── 2) Calcula estadísticas con lifelines
     ├── 3) Construye un prompt limpio (~200 tokens)
     ├── 4) Llama a generate_explanation(prompt, idioma)
     │
     ▼
 FUNCIÓN generate_explanation() en ollama_AI.py
     │
     ├── 5) Prepara system prompt + instrucción de idioma
     ├── 6) Envía petición HTTP POST a 127.0.0.1:8080
     │
     ▼
 LLAMA-SERVER (proceso separado)
     │
     ├── 7) Qwen2.5 genera la respuesta en CPU
     ├── 8) Devuelve JSON con la respuesta
     │
     ▼
 DE VUELTA en ollama_AI.py
     │
     ├── 9) Extrae el texto de la respuesta
     ├── 10) Lo devuelve al callback
     │
     ▼
 El usuario ve la explicación en el Textarea del dashboard
```

### 2.10 Parámetros clave

| Parámetro | Valor | Qué hace |
|---|---|---|
| `temperature` | 0.2 | Respuestas precisas y consistentes |
| `max_tokens` | 1200 | Máximo ~600 palabras en la respuesta |
| `timeout` | 600s | Cancela si tarda >10 minutos |
| `-t 8` | 8 threads | Núcleos de CPU usados para inferencia |
| `-ngl 0` | 0 capas GPU | Todo se ejecuta en CPU |

### 2.11 Ejecución

Se necesitan **2 terminales simultáneas**:

```bash
# Terminal 1: Servidor de IA (esperar a que diga "listening on 127.0.0.1:8080")
.\START_LLAMA_SERVER.bat

# Terminal 2: Dashboard
python cargaDataset.py
```

**IMPORTANTE**: No es necesario tener la pestaña `http://127.0.0.1:8080` abierta en el navegador. Esa es solo una interfaz de prueba. El dashboard se comunica directamente con el servidor por HTTP.
