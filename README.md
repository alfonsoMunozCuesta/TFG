# Ampliacion y mejora de un dashboard de analisis de supervivencia

Este repositorio contiene el codigo fuente y la documentacion de un Trabajo Fin de Grado centrado en la ampliacion de una aplicacion web interactiva para el analisis de supervivencia aplicado al abandono universitario.

La aplicacion esta desarrollada en Python y Dash. Permite cargar y preprocesar datos academicos, ejecutar distintas tecnicas de analisis de supervivencia, visualizar resultados, generar interpretaciones asistidas por inteligencia artificial local y exportar informes en PDF.

## Funcionalidades principales

- Carga, validacion y preprocesamiento de un dataset academico.
- Visualizacion del dataset limpio y analisis descriptivo de covariables.
- Ejecucion de tecnicas de supervivencia:
  - Kaplan-Meier.
  - Regresion de Cox.
  - Test de Log-Rank.
  - Modelo Exponencial.
  - Modelo de Weibull.
  - Random Survival Forest.
- Comparacion orientativa entre tecnicas.
- Soporte multilingue en castellano e ingles.
- Generacion de explicaciones mediante un modelo local de IA.
- Exportacion de resultados, tablas, graficas e interpretaciones en PDF.

## Organizacion del repositorio

```text
.
|-- assets/                         Recursos visuales y estilos del dashboard
|-- benchmarks/                     Resultados CSV/JSON de mediciones de IA
|-- data/                           Datos de entrada utilizados por la aplicacion
|-- doc/memoria_tfg/                Memoria del TFG en LaTeX
|-- downloads/                      Informes PDF generados por la aplicacion
|-- models/                         Ubicacion prevista para el modelo GGUF local
|-- tools/                          Ubicacion prevista para llama-server.exe
|-- analysis_callbacks.py           Callbacks principales de analisis e interaccion
|-- cargaDataset.py                 Punto de entrada de la aplicacion Dash
|-- config.py                       Configuracion de rutas y servidor de IA
|-- layout.py                       Definicion de paginas y componentes visuales
|-- preprocesamiento.py             Limpieza y preparacion del dataset
|-- kaplan_meier.py                 Curvas Kaplan-Meier
|-- cox_regression.py               Regresion de Cox y visualizaciones asociadas
|-- log_rank_test.py                Test Log-Rank y curvas comparativas
|-- exponential.py                  Modelo Exponencial
|-- weibull.py                      Modelo de Weibull
|-- rsf.py                          Random Survival Forest
|-- survival_plots.py               Graficas estadisticas reutilizables
|-- ollama_AI.py                    Integracion con el modelo local de IA
|-- pdf_callbacks.py                Callbacks de exportacion a PDF
|-- pdf_exporter.py                 Construccion de informes PDF
|-- translations.py                 Textos de interfaz en espanol e ingles
|-- benchmark_ia.py                 Benchmark del modulo IA del sistema actual
|-- benchmark_ia_lucia.py           Benchmark comparativo del TFG base
|-- START_DASH_APP.bat              Arranque rapido del dashboard en Windows
`-- START_LLAMA_SERVER.bat          Arranque rapido del servidor local de IA
```

## Modulos principales

| Archivo | Responsabilidad |
| --- | --- |
| `cargaDataset.py` | Inicializa Dash, registra callbacks y arranca el servidor web local. |
| `layout.py` | Define la estructura visual de las paginas del dashboard. |
| `analysis_callbacks.py` | Conecta la interfaz con los modelos, las tablas, las graficas y la IA. |
| `preprocesamiento.py` | Valida el CSV, reduce el dataset y construye una fila representativa por estudiante. |
| `kaplan_meier.py`, `cox_regression.py`, `log_rank_test.py` | Implementan las tecnicas clasicas de supervivencia. |
| `exponential.py`, `weibull.py`, `rsf.py` | Implementan las tecnicas anadidas en la ampliacion del TFG. |
| `ollama_AI.py` | Construye prompts academicos y llama al servidor local compatible con API OpenAI. |
| `pdf_exporter.py`, `pdf_callbacks.py` | Generan informes PDF a partir de los resultados seleccionados por el usuario. |
| `translations.py` | Centraliza textos traducibles para la interfaz en castellano e ingles. |

El codigo esta documentado con comentarios y docstrings para facilitar su revision. Este README actua como guia general de orientacion, no como manual exhaustivo de codigo fuente.

## Requisitos

Entorno utilizado durante el desarrollo y las pruebas:

- Windows 11.
- Python 3.13.2.
- Dash 3.1.0.
- Librerias listadas en `requirements.txt`.

Instalacion recomendada:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecucion del dashboard

Desde la raiz del proyecto:

```powershell
python cargaDataset.py
```

Tambien puede ejecutarse con el script incluido:

```powershell
.\START_DASH_APP.bat
```

Despues, abrir en el navegador:

```text
http://127.0.0.1:8050
```

## Modulo de inteligencia artificial local

La IA es opcional. El analisis estadistico funciona aunque el servidor de IA no este activo.

Para generar explicaciones automaticas se utiliza un servidor local compatible con la API de OpenAI, ejecutado mediante `llama-server`. La configuracion por defecto espera:

```text
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
tools/llama.cpp/llama-server.exe
```

Estos archivos pueden no estar incluidos en GitHub por tamano o por tratarse de recursos externos. En ese caso deben colocarse manualmente en esas rutas o configurar las variables `MODEL_PATH` y `LLAMA_SERVER`.

Arranque del servidor de IA:

```powershell
.\START_LLAMA_SERVER.bat
```

Endpoint utilizado por defecto:

```text
http://127.0.0.1:8000/v1/chat/completions
```

## Dataset esperado

La aplicacion trabaja con un CSV academico que debe contener, como minimo, las columnas necesarias para el analisis:

- `id_student`
- `date`
- `final_result`
- `gender_F`
- `disability_N`
- `age_band_0-35`
- `age_band_35-55`
- `age_band_55<=`
- `highest_education_A Level or Equivalent`
- `highest_education_HE Qualification`
- `highest_education_Lower Than A Level`
- `highest_education_No Formal quals`
- `highest_education_Post Graduate Qualification`
- `studied_credits`

El preprocesamiento valida la estructura, transforma el resultado final en evento de abandono, consolida los registros por estudiante y genera el dataset limpio utilizado por los modelos.

## Flujo de uso

1. Iniciar el dashboard.
2. Cargar un CSV valido.
3. Ejecutar el preprocesamiento.
4. Revisar el dataset limpio.
5. Acceder a los modulos de analisis:
   - analisis de covariables,
   - Kaplan-Meier,
   - Cox,
   - Log-Rank,
   - Exponencial,
   - Weibull,
   - Random Survival Forest,
   - comparacion de tecnicas.
6. Consultar tablas, metricas y graficas.
7. Opcionalmente, solicitar interpretacion mediante IA.
8. Exportar los resultados a PDF.

## Documentacion academica

La memoria del TFG esta en:

```text
doc/memoria_tfg/
```

Archivo principal:

```text
doc/memoria_tfg/__memoria.tex
```

Compilacion en Windows:

```powershell
cd doc\memoria_tfg
.\compilar_memoria.ps1
```

El PDF generado se guarda como:

```text
doc/memoria_tfg/__memoria.pdf
```

El manual de usuario esta incluido como apendice dentro de la memoria, en el archivo `manual_de_usuario.tex`.

## Pruebas y benchmark

El capitulo de pruebas de la memoria documenta la validacion funcional de los casos de uso principales. Ademas, se incluyen scripts para medir el rendimiento del modulo de IA:

```powershell
python benchmark_ia.py
python benchmark_ia_lucia.py
```

Los resultados se almacenan en:

```text
benchmarks/
```

Estas mediciones comparan tiempos de respuesta y consumo aproximado de recursos en el equipo de pruebas. No deben interpretarse como una comparacion universal entre modelos, sino como una evaluacion practica de las configuraciones utilizadas en el TFG.

## Notas para revision del tribunal

- La aplicacion se ejecuta en local y no requiere desplegar un servidor externo.
- El modulo de IA es auxiliar: ayuda a interpretar resultados, pero no sustituye el analisis estadistico.
- El sistema esta planteado como una herramienta de apoyo exploratorio para estudiar patrones asociados al abandono academico, no como un sistema automatico de decision.
- La memoria en LaTeX documenta objetivos, antecedentes, requisitos, diseno, implementacion, pruebas, limitaciones y conclusiones.
