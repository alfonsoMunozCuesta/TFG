import dash
from dash import Dash, dcc, html, dash_table
import pandas as pd
from pathlib import Path
import dash_daq as daq
from dash.dependencies import Input, Output, State
import base64
import plotly.express as px
import io
from layout import (
    create_survival_analysis_page, create_covariate_analysis_page, 
    create_kaplan_meier_page, create_cox_regression_page, 
    create_log_rank_page, create_ver_dataset_page, 
    create_weibull_analysis_page, create_exponential_analysis_page,
    create_rsf_analysis_page,
    display_logrank_summary_table
)
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc
from cox_regression import run_cox_regression
from log_rank_test import perform_log_rank_test
from survival_plots import plot_logrank_curves, plot_cox_hazard_ratios
from preprocesamiento import preprocess_data
from weibull import build_weibull_analysis
from exponential import build_exponential_analysis
from rsf import build_rsf_analysis, build_rsf_profile_analysis
import matplotlib.pyplot as plt
import requests
from ollama_AI import generate_explanation, generate_interpretation_for_pdf
import plotly.graph_objs as go
from dash import callback_context
from dash.exceptions import PreventUpdate
from translations import get_translation
from pdf_callbacks import register_pdf_export_callbacks  # PDF export HABILITADO
from config import LLAMA_SERVER_URL, MODEL_NAME

# Inicializar la aplicación Dash
app = Dash(__name__, suppress_callback_exceptions=True)
app.config['suppress_callback_exceptions'] = True

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent
TEMP_DATA_PATH = BASE_DIR / 'data' / 'temp_data.csv'
CLEAN_DATA_PATH = BASE_DIR / 'dataset_limpio.csv'

# Inicializar como None - se cargarán cuando sea necesario
df = None
df_limpio = None

def load_dataframes():
    """Carga los dataframes bajo demanda si no existen."""
    global df, df_limpio
    if df is None:
        try:
            df = pd.read_csv(TEMP_DATA_PATH, sep=';')
        except FileNotFoundError:
            print("⚠️  Advertencia: temp_data.csv no encontrado")
    if df_limpio is None:
        try:
            df_limpio = pd.read_csv(CLEAN_DATA_PATH, sep=';')
        except FileNotFoundError:
            print("⚠️  Advertencia: dataset_limpio.csv no encontrado")


def _read_split_json(json_value):
    """Lee JSON con orient='split' sin depender de cadenas literales."""
    if json_value is None:
        return None
    if isinstance(json_value, str):
        return pd.read_json(io.StringIO(json_value), orient='split')
    return pd.read_json(json_value, orient='split')


def _humanize_label(value):
    """Convierte etiquetas técnicas a texto legible para interpretación narrativa."""
    if value is None:
        return "N/A"

    text = str(value)
    replacements = {
        'highest_education_': 'education_',
        'age_band_': 'age_',
        'gender_F': 'Female',
        'disability_N': 'No disability',
        '_': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def _looks_like_list_output(text):
    """Detecta respuestas tipo lista para relanzar reescritura en prosa."""
    if not text:
        return False
    stripped = text.strip()
    if stripped.startswith('-') or stripped.startswith('*'):
        return True
    list_markers = ('1)', '2)', '3)', '1.', '2.', '3.')
    return any(marker in stripped for marker in list_markers)


def _dataset_signature_from_json(df_json):
    """Genera una firma estable para detectar cambios de dataset en el flujo."""
    if not df_json:
        return ""
    return str(hash(df_json))

# ==================== FUNCIÓN DE IA ====================
def responder_pregunta_con_llama3(pregunta: str, language: str = 'es') -> str:
    """
    Envía la pregunta a llama-server (compatibilidad OpenAI API).
    Si el servidor no está disponible, devuelve un mensaje amigable.
    """
    import time
    
    try:
        # Payload compatible con OpenAI API (formato que usa llama-server)
        system_prompt = (
            "You are a statistical analyst for a survival-analysis bachelor's thesis. "
            "Write in clear and professional academic style, in 2-3 short paragraphs, "
            "without numbered lists or bullet points. "
            "Do not invent data; use only the provided results. "
            "Write strictly in English."
            if language == 'en' else
            "Eres un analista estadístico para un TFG de supervivencia. "
            "Redacta en estilo académico claro y profesional, en 2-3 párrafos breves, "
            "sin listas numeradas ni viñetas. "
            "No inventes datos; usa solo los resultados proporcionados. "
            "Escribe estrictamente en español."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": pregunta}
            ],
            "temperature": 0.15,
            "max_tokens": 320
        }
        
        # Registro de inicio
        inicio = time.time()
        
        # Realiza la solicitud HTTP al servidor llama-server (timeout de 10 minutos para permitir procesamiento)
        response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=600)
        response.raise_for_status()
        
        # Calcular tiempo de respuesta
        tiempo_respuesta = time.time() - inicio
        
        # Extrae la respuesta generada (formato OpenAI)
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()

        if _looks_like_list_output(content):
            rewrite_system_prompt = (
                "Rewrite in bachelor's-thesis academic style, in prose, 2 short paragraphs, "
                "without numbered lists or bullet points. Keep exactly the same factual content. "
                "Write strictly in English."
                if language == 'en' else
                "Reescribe en estilo académico de TFG, en prosa, 2 párrafos breves, "
                "sin listas numeradas ni viñetas. Mantén exactamente el contenido factual. "
                "Escribe estrictamente en español."
            )

            rewrite_user_prompt = (
                f"Rewrite this text in academic prose without lists:\n\n{content}"
                if language == 'en' else
                f"Reescribe en prosa académica este texto sin listas:\n\n{content}"
            )

            rewrite_payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": rewrite_system_prompt
                    },
                    {
                        "role": "user",
                        "content": rewrite_user_prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 320,
                "stream": False
            }
            rewrite_response = requests.post(LLAMA_SERVER_URL, json=rewrite_payload, timeout=600)
            rewrite_response.raise_for_status()
            rewrite_result = rewrite_response.json()
            rewritten = rewrite_result['choices'][0]['message']['content'].strip()
            if rewritten:
                content = rewritten
        
        if not content:
            raise ValueError("No se recibió una respuesta válida del modelo.")
        
        return content
        
    except requests.exceptions.ConnectionError:
        return ""
    except requests.exceptions.Timeout:
        return ""
    except Exception as e:
        print(f"Error consultando Qwen2.5: {e}")
        return ""


def _build_km_interpretation_context(df, variable_actual):
    """Construye un resumen interpretable de la curva Kaplan-Meier actual."""
    if df is None or variable_actual not in df.columns:
        return ""

    from lifelines import KaplanMeierFitter

    context_lines = []
    kmf = KaplanMeierFitter()

    if variable_actual in ['gender_F', 'disability_N']:
        label_map = {
            'gender_F': {1: 'Femenino', 0: 'Masculino'},
            'disability_N': {1: 'Con discapacidad', 0: 'Sin discapacidad'}
        }
        for group_value in sorted(df[variable_actual].dropna().unique()):
            group_df = df[df[variable_actual] == group_value]
            if len(group_df) == 0:
                continue
            kmf.fit(group_df['date'], event_observed=group_df['final_result'])
            final_survival = float(kmf.survival_function_.iloc[-1, 0]) if len(kmf.survival_function_) > 0 else 0.0
            context_lines.append(
                f"- {label_map.get(variable_actual, {}).get(group_value, group_value)}: "
                f"n={len(group_df)}, eventos={int(group_df['final_result'].sum())}, "
                f"supervivencia final≈{final_survival:.3f}"
            )

    elif variable_actual in ['age_band', 'highest_education']:
        mapping = {
            'age_band': ['age_band_0-35', 'age_band_35-55', 'age_band_55<='],
            'highest_education': [
                'highest_education_A Level or Equivalent',
                'highest_education_HE Qualification',
                'highest_education_Lower Than A Level',
                'highest_education_Post Graduate Qualification'
            ]
        }
        for col in mapping.get(variable_actual, []):
            if col not in df.columns:
                continue
            group_df = df[df[col] == 1]
            if len(group_df) == 0:
                continue
            kmf.fit(group_df['date'], event_observed=group_df['final_result'])
            final_survival = float(kmf.survival_function_.iloc[-1, 0]) if len(kmf.survival_function_) > 0 else 0.0
            label = col.split('_', 1)[1] if '_' in col else col
            context_lines.append(
                f"- {label}: n={len(group_df)}, eventos={int(group_df['final_result'].sum())}, "
                f"supervivencia final≈{final_survival:.3f}"
            )

    elif variable_actual == 'studied_credits' and 'studied_credits' in df.columns:
        df_temp = df[['studied_credits', 'date', 'final_result']].dropna()
        if len(df_temp) > 0:
            n_unique = df_temp['studied_credits'].nunique()
            if n_unique <= 5:
                unique_vals = sorted(df_temp['studied_credits'].unique())
                value_to_group = {val: i for i, val in enumerate(unique_vals)}
                df_temp['group'] = df_temp['studied_credits'].map(value_to_group)
                groups = sorted(df_temp['group'].unique())
                labels = [f"Grupo {i+1} ({unique_vals[i]:.0f} créditos)" for i in groups]
            else:
                df_temp['group'] = pd.qcut(df_temp['studied_credits'], q=5, labels=False, duplicates='drop')
                groups = sorted(df_temp['group'].unique())
                labels = [f"Quintil {i+1}" for i in groups]

            for i, group_val in enumerate(groups):
                group_df = df_temp[df_temp['group'] == group_val]
                if len(group_df) == 0:
                    continue
                kmf.fit(group_df['date'], event_observed=group_df['final_result'])
                final_survival = float(kmf.survival_function_.iloc[-1, 0]) if len(kmf.survival_function_) > 0 else 0.0
                context_lines.append(
                    f"- {labels[i]}: n={len(group_df)}, eventos={int(group_df['final_result'].sum())}, "
                    f"supervivencia final≈{final_survival:.3f}"
                )

    return "\n".join(context_lines)


def _build_logrank_interpretation_context(logrank_store_data):
    """Convierte los resultados de Log-Rank en texto interpretable."""
    if not isinstance(logrank_store_data, dict):
        return ""

    result_blocks = []

    # Formato heredado: {'results_json': '...'}
    if logrank_store_data.get('results_json'):
        result_blocks.append(logrank_store_data.get('results_json'))

    # Formato actual: {'results': [{'results_json': '...'}, ...]}
    for item in logrank_store_data.get('results', []) or []:
        if isinstance(item, dict) and item.get('results_json'):
            result_blocks.append(item.get('results_json'))

    if not result_blocks:
        return ""

    lines = []
    for block in result_blocks:
        try:
            results_df = _read_split_json(block)
        except Exception:
            continue

        if results_df is None or results_df.empty:
            continue

        for _, row in results_df.head(8).iterrows():
            p_value = row.get('p_value', row.get('p', None))
            stat = row.get('test_statistic', row.get('chi2', None))
            conclusion = row.get('Conclusión', '')
            group_a = _humanize_label(row.get('Grupo A', 'N/A'))
            group_b = _humanize_label(row.get('Grupo B', 'N/A'))
            decision = row.get('Decisión', '')
            lines.append(
                f"- {_humanize_label(row.get('Covariable', row.get('Variable', 'N/A')))} | {group_a} vs {group_b}: chi2={stat}, p={p_value}, decisión={decision}, conclusión={conclusion}"
            )

    return "\n".join(lines)


# Barra de navegación fija en la parte superior
navbar = html.Div([
    html.Div([ 
        html.Div([
            html.Button(get_translation('es', 'navbar_home'), id='inicio-btn', n_clicks=0, className='navbar-link', style={'border': 'none', 'background': 'none', 'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '14px'}),
            dcc.Link(get_translation('es', 'navbar_view_dataset'), href='/ver-dataset', className='navbar-link', id='navbar-view-dataset'), 
            dcc.Link(get_translation('es', 'navbar_covariate_analysis'), href='/covariate-analysis', className='navbar-link', id='navbar-covariate-analysis'),
            dcc.Link(get_translation('es', 'navbar_survival_analysis'), href='/survival-analysis', className='navbar-link', id='navbar-survival-analysis'),
        ], className='navbar-links', style={'flex': '1'}),
        # Selector de idioma con banderas + switch en la esquina derecha
        html.Div([
            html.Img(
                src='/assets/spain.jpg',
                alt='Español',
                style={
                    'width': '28px',
                    'height': '20px',
                    'objectFit': 'cover',
                    'borderRadius': '3px',
                    'boxShadow': '0 1px 4px rgba(0,0,0,0.2)'
                }
            ),
            daq.ToggleSwitch(
                id='language-toggle',
                value=False,
                size=36,
                color='#4a4a4a',
                style={'margin': '0 10px'}
            ),
            html.Img(
                src='/assets/ingles.png',
                alt='English',
                style={
                    'width': '28px',
                    'height': '20px',
                    'objectFit': 'cover',
                    'borderRadius': '3px',
                    'boxShadow': '0 1px 4px rgba(0,0,0,0.2)'
                }
            )
        ], className='language-switch-wrap', style={'marginRight': '42px', 'display': 'flex', 'alignItems': 'center', 'gap': '2px'})
    ], className='navbar-inner', style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'width': '100%'})
], id='navbar', style={'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'})

app.layout = html.Div([ 
    navbar, 
    dcc.Location(id='url', refresh=False),  
    dcc.Store(id='language-store', data='es'),  # Store para almacenar el idioma actual
    dcc.Store(id='df-store', data=None),  # Store para almacenar el dataframe procesado
    dcc.Store(id='dataset-signature-store', data=''),  # Firma del dataset para invalidar análisis obsoletos
    dcc.Store(id='km-current-variable', data=''),  # Store para rastrear variable actual en Kaplan-Meier
    dcc.Store(id='cox-current-variables', data=''),  # Store para rastrear variables en Cox Regression
    dcc.Store(id='logrank-current-variable', data=''),  # Store para rastrear variable en Log-Rank
    dcc.Store(id='logrank-selected-covariables', data=[]),  # Store para covariables seleccionadas en Log-Rank
    dcc.Store(id='cox-selected-covariables', data=[]),  # Store para covariables seleccionadas en Cox
    dcc.Store(id='logrank-test-output-store', data=None),  # Store para resultados de Log-Rank
    dcc.Store(id='cox-regression-output-store', data=None),  # Store para resultados de Cox
    dcc.Store(id='cox-figure-store', data=None),  # Store para gráfica Forest Plot de Cox
    dcc.Store(id='logrank-figure-store', data=None),  # Store para gráfica de Log-Rank
    dcc.Store(id='weibull-ai-text-store', data=''),  # Store para guardar explicación IA Weibull
    dcc.Store(id='exponential-ai-text-store', data=''),  # Store para guardar explicación IA Exponencial
    dcc.Store(id='weibull-ai-language-store', data=''),  # Idioma en que se generó la IA de Weibull
    dcc.Store(id='exponential-ai-language-store', data=''),  # Idioma en que se generó la IA de Exponencial
    html.Div(id='page-content'),  
    # Cuadro de confirmación
    dcc.ConfirmDialog(
        id='confirm-dialog',
        message='¿Estás seguro de que deseas volver a la página inicial? Perderás el dataset cargado y todo el análisis realizado.',
        displayed=False,  # Inicialmente no se muestra
        submit_n_clicks=0,  # Mantener el contador de clicks
        cancel_n_clicks=0  
    )
])
# Callback para almacenar el idioma seleccionado
@app.callback(
    Output('language-store', 'data'),
    Input('language-toggle', 'value')
)
def update_language(toggle_value):
    return 'en' if toggle_value else 'es'


@app.callback(
    [Output('inicio-btn', 'children'),
     Output('navbar-view-dataset', 'children'),
     Output('navbar-covariate-analysis', 'children'),
     Output('navbar-survival-analysis', 'children')],
    Input('language-store', 'data')
)
def update_navbar_labels(language):
    return (
        get_translation(language, 'navbar_home'),
        get_translation(language, 'navbar_view_dataset'),
        get_translation(language, 'navbar_covariate_analysis'),
        get_translation(language, 'navbar_survival_analysis'),
    )

@app.callback(
    Output('confirm-dialog', 'displayed'),
    Input('inicio-btn', 'n_clicks'),
    prevent_initial_call=True  # Evita que se active al cargar la página
)
def mostrar_confirmacion(n_clicks):
    if n_clicks:
        return True 
    return False

def _create_dataset_locked_message(language):
    return html.Div(
        [
            html.Div(
                [
                    html.H2("❌ " + get_translation(language, 'error_preprocess_title'), style={'marginBottom': '12px', 'color': '#b03a2e'}),
                    html.P(get_translation(language, 'error_dataset_not_loaded'), style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P(
                        (
                            'First click "Preprocess CSV" on the upload page to enable this section.'
                            if language == 'en' else
                            'Primero pulsa "Preprocesar CSV" desde la página de carga para habilitar esta sección.'
                        ),
                        style={'fontSize': '15px', 'color': '#555', 'marginBottom': '0'}
                    )
                ],
                style={
                    'maxWidth': '900px',
                    'margin': '90px auto 40px auto',
                    'padding': '28px',
                    'backgroundColor': '#fff5f5',
                    'border': '2px solid #f5c6cb',
                    'borderRadius': '14px',
                    'boxShadow': '0 10px 30px rgba(176, 58, 46, 0.08)',
                    'textAlign': 'center'
                }
            )
        ]
    )

@app.callback(
    Output('confirm-dialog', 'message'),
    Input('language-store', 'data')
)
def update_confirm_message(language):
    return get_translation(language, 'confirmar_inicio')


# Callbacks para manejar la navegación entre páginas
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('language-store', 'data')],
    [State('df-store', 'data')]
)
def display_page(pathname, language, df_json):
    dataset_loaded = df_json is not None

    if pathname == '/':
        return create_home_page(language)

    if pathname in ['/ver-dataset', '/covariate-analysis', '/survival-analysis', '/survival-analysis/kaplan-meier', '/survival-analysis/cox-regression', '/survival-analysis/log-rank', '/survival-analysis/weibull', '/survival-analysis/exponential', '/survival-analysis/rsf'] and not dataset_loaded:
        return _create_dataset_locked_message(language)

    if pathname == '/covariate-analysis':
        return create_covariate_analysis_page(language)
    elif pathname == '/survival-analysis':
        return create_survival_analysis_page(language)
    elif pathname == '/survival-analysis/kaplan-meier':
        return create_kaplan_meier_page(language)
    elif pathname == '/survival-analysis/cox-regression':
        return create_cox_regression_page(language)
    elif pathname == '/survival-analysis/log-rank':
        return create_log_rank_page(language)
    elif pathname == '/survival-analysis/weibull':
        return create_weibull_analysis_page(language)
    elif pathname == '/survival-analysis/exponential':
        return create_exponential_analysis_page(language)
    elif pathname == '/survival-analysis/rsf':
        return create_rsf_analysis_page(language)
    elif pathname == '/ver-dataset':
        return create_ver_dataset_page(language)
    else:
        return create_home_page(language)


@app.callback(
    Output('url', 'pathname'),
    [Input('confirm-dialog', 'submit_n_clicks'),
     Input('confirm-dialog', 'cancel_n_clicks')],
    prevent_initial_call=True
)
def navegar_a_inicio(submit_n_clicks, cancel_n_clicks):

    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id']
    if 'submit_n_clicks' in trigger_id:
        print("Aceptar clickeado")
        return '/'
    
    if 'cancel_n_clicks' in trigger_id:
        print("Cancelar clickeado")
        return dash.no_update
    
    return dash.no_update

# Página de HOME - como función para usar traducciones dinámicamente
def create_home_page(language='es'):
    return html.Div([
        html.Div([ 
            html.Video(
                src='/assets/banner.mp4', 
                id='banner-video', 
                autoPlay=True, 
                muted=True, 
                loop=True, 
                style={
                    'width': '100%', 
                    'maxHeight': '350px', 
                    'display': 'block', 
                    'marginTop': '0px', 
                    'marginBottom': '0px',
                    'objectFit': 'cover'  
                }
            )
        ], id="banner-container", style={'width': '100%', 'padding': '0', 'margin': '0'}),

        html.Div([
            html.Div([
                html.H1(get_translation(language, 'dashboard_title'), className='home-title-formal', style={'textAlign': 'center'}),
                html.P(
                    'Entorno de análisis estadístico para el estudio de abandono y supervivencia académica.' if language == 'es'
                    else 'Statistical analysis environment for dropout and academic survival studies.',
                    className='home-subtitle-formal'
                ),

                dcc.Loading(
                    id="loading-spinner",
                    type="circle",
                    children=html.Div([
                        html.H3(get_translation(language, 'cargar_dataset'), id='upload-text', className='home-upload-title'),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Button(get_translation(language, 'sube_csv'), className='home-upload-btn'),
                            multiple=False
                        ),
                        html.Div(id='output-data-upload', className='home-upload-feedback')
                    ], className='home-upload-block')
                ),

                html.Div([
                    html.Button(get_translation(language, 'preprocesa_csv'), id='load-clean', n_clicks=0, className='home-preprocess-btn'),
                ], className='home-preprocess-block')
            ], className='home-main-panel')
        ], style={'padding': '10px 0 30px 0'}),
    ])

# Página inicial
home_page = create_home_page('es')
#ocultar frase incial: "Cargar Dataset..."
@app.callback(
    Output('upload-text', 'style'),  # Cambiar el estilo del texto
    [Input('upload-data', 'contents')],
    prevent_initial_call=True
)
def hide_upload_text(contents):
    # Si se ha cargado un archivo, ocultamos el texto
    if contents is not None:
        return {'display': 'none'}  # Ocultar el texto
    return {'display': 'block'} 

# Función para procesar el archivo cargado y mostrarlo en una tabla
def display_data(df, title):
    return html.Div([
        html.H5(title),
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": col, "id": col} for col in df.columns],  
            data=df.to_dict('records'),  
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'}, 
        ),
    ])
# Función para cargar el archivo CSV
def parse_contents(contents):
    """
    Decodifica y carga un archivo CSV desde un upload de Dash.
    Intenta automáticamente detecatar el separador correcto.
    
    Args:
        contents: String codificado en base64 del archivo
        
    Returns:
        DataFrame con los datos cargados
        
    Raises:
        ValueError: Si no se puede procesar el archivo
    """
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Intentar decodificar con UTF-8
        try:
            csv_string = decoded.decode('utf-8')
        except UnicodeDecodeError:
            # Intentar con latin-1 como fallback
            try:
                csv_string = decoded.decode('latin-1')
            except:
                raise ValueError("No se pudo decodificar el archivo. Verifica la codificación (UTF-8 o Latin-1)")
        
        # Intentar cargar con separador ";" primero (esperado)
        try:
            df = pd.read_csv(io.StringIO(csv_string), sep=";")
            return df
        except Exception as e1:
            # Intentar con "," como fallback
            try:
                df = pd.read_csv(io.StringIO(csv_string), sep=",")
                print(f"⚠️  Advertencia: Se detectó separador ',' en lugar de ';'")
                return df
            except Exception as e2:
                raise ValueError(f"No se pudo leer el CSV. Verifica el formato y separador.")
                
    except Exception as e:
        raise ValueError(f"Error procesando el archivo: {str(e)}")

def verificar_archivo_correcto(contents, filename):
    # Compara el nombre del archivo cargado con el archivo esperado
    archivo_esperado = "temp_data.csv"
    
    # Verificar si el nombre del archivo cargado es el esperado
    if filename != archivo_esperado:
        return False
    return True


# Función para actualizar la página y mostrar el archivo cargado
@app.callback(
    [Output('upload-data', 'style'),
     Output('load-clean', 'style'), 
     Output('output-data-upload', 'children'),
    Output('df-store', 'data'),
    Output('dataset-signature-store', 'data')], 
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'), 
     Input('load-clean', 'n_clicks')],
    [State('language-store', 'data'),
     State('df-store', 'data')]
)

def update_output(contents, filename, n_clicks, language, current_df_json):
    try:
        if language not in ['es', 'en']:
            language = 'es'

        safe_clicks = n_clicks or 0

        if contents is None:
            # Si ya hay un dataset preprocesado en memoria, mantenerlo sin forzar recarga
            if current_df_json is not None:
                try:
                    df_cached = _read_split_json(current_df_json)
                    return (
                        {'display': 'none'},
                        {'display': 'none'},
                        display_data(df_cached, get_translation(language, 'archivo_preprocesado')),
                        current_df_json,
                        _dataset_signature_from_json(current_df_json),
                    )
                except Exception:
                    # Si falla la reconstrucción, limpiar estado corrupto de forma segura
                    return (
                        {'display': 'block'},
                        {'display': 'none'},
                        html.Div([get_translation(language, 'no_archivo_cargado')], style={'marginTop': '20px', 'marginBottom': '0px'}),
                        None,
                        '',
                    )

            return {'display': 'block'}, {'display': 'none'}, html.Div([get_translation(language, 'no_archivo_cargado')], style={'marginTop': '20px', 'marginBottom': '0px'}), None, ''

        if not verificar_archivo_correcto(contents, filename):
            return {'display': 'none'}, {'display': 'none'}, html.Div(
                [
                    get_translation(language, 'error_archivo'),
                    html.Br(),
                    get_translation(language, 'archivo_correcto')
                ],
                style={
                    'color': 'red',
                    'fontSize': '20px',
                    'textAlign': 'center',
                    'marginTop': '20px'
                }
            ), None, ''

        # Cargar el archivo CSV con manejo de errores
        try:
            df = parse_contents(contents)
        except Exception as e:
            return {'display': 'none'}, {'display': 'none'}, html.Div(
                [
                    html.H3(f"❌ {get_translation(language, 'error_loading_csv_title')}"),
                    html.P(get_translation(language, 'error_loading_csv_body')),
                    html.Ul([
                        html.Li(get_translation(language, 'error_loading_csv_tip_1')),
                        html.Li(get_translation(language, 'error_loading_csv_tip_2')),
                        html.Li(get_translation(language, 'error_loading_csv_tip_3'))
                    ]),
                    html.P(f"Detalles: {str(e)}")
                ],
                style={
                    'color': 'red',
                    'fontSize': '16px',
                    'padding': '20px',
                    'border': '2px solid red',
                    'borderRadius': '5px',
                    'marginTop': '20px'
                }
            ), None, ''

        if safe_clicks > 0:
            # Ejecutar el preprocesamiento con manejo de errores
            try:
                df_procesado = preprocess_data(df)
                processed_json = df_procesado.to_json(date_format='iso', orient='split')
                return {'display': 'none'}, {'display': 'none'}, display_data(df_procesado, get_translation(language, 'archivo_preprocesado')), processed_json, _dataset_signature_from_json(processed_json)
            except (ValueError, TypeError) as e:
                return {'display': 'none'}, {'display': 'none'}, html.Div(
                    [
                        html.H3(f"❌ {get_translation(language, 'error_preprocess_title')}"),
                        html.P(str(e)),
                        html.Hr(),
                        html.P(get_translation(language, 'error_preprocess_body'))
                        , html.Ul([
                            html.Li("id_student"),
                            html.Li("date"),
                            html.Li("final_result"),
                            html.Li("gender_F, disability_N"),
                            html.Li("age_band_* (0-35, 35-55, 55<=)"),
                            html.Li("highest_education_* (5 tipos)"),
                            html.Li("studied_credits")
                        ])
                    ],
                    style={
                        'color': 'red',
                        'fontSize': '14px',
                        'padding': '20px',
                        'border': '2px solid red',
                        'borderRadius': '5px',
                        'marginTop': '20px',
                        'backgroundColor': '#fff5f5'
                    }
                ), None, ''
            except Exception as e:
                return {'display': 'none'}, {'display': 'none'}, html.Div(
                    [
                        html.H3(f"❌ {get_translation(language, 'error_unexpected_title')}"),
                        html.P(get_translation(language, 'error_unexpected_body')),
                        html.P(f"Detalles: {str(e)}"),
                        html.P(get_translation(language, 'error_unexpected_contact'))
                    ],
                    style={
                        'color': 'red',
                        'fontSize': '14px',
                        'padding': '20px',
                        'border': '2px solid red',
                        'borderRadius': '5px',
                        'marginTop': '20px',
                        'backgroundColor': '#fff5f5'
                    }
                ), None, ''

        # Si no se ha presionado el botón de limpiar, mostrar el archivo bruto
        return {'display': 'none'}, {'display': 'inline-block'}, display_data(df, get_translation(language, 'archivo_bruto')), None, ''

    except Exception as e:
        print(f"[UPDATE_OUTPUT] Error no controlado: {e}")
        return (
            {'display': 'block'},
            {'display': 'none'},
            html.Div(
                f"Error interno del servidor: {str(e)}" if language == 'es' else f"Internal server error: {str(e)}",
                style={'color': '#b03a2e', 'fontWeight': 'bold', 'marginTop': '20px'}
            ),
            current_df_json,
            _dataset_signature_from_json(current_df_json)
        )


#maneja que no aparezca la barra de navegacion hasta que se limpie el dataset
@app.callback(
    Output('navbar', 'style'),
    [Input('df-store', 'data')]
)
def toggle_navbar(df_json):
    if df_json is not None:
        return {'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'}
    return {'display': 'none'}  # Si no se ha presionado el botón, la barra de navegación permanece oculta

#maneja navegacion de kaplan
@app.callback(
    [Output('km-cov-div', 'children'),
     Output('km-current-variable', 'data')],
    [Input('botonG', 'n_clicks'), Input('botonDisc', 'n_clicks'), 
     Input('botonAge', 'n_clicks'), Input('botonEdu', 'n_clicks'), 
     Input('botonCredits', 'n_clicks'), Input('botonNone', 'n_clicks')],
    [State('df-store', 'data'),
     State('language-store', 'data')],
    prevent_initial_call=True
)
def update_km_cov(gender_clicks, disability_clicks, age_clicks, edu_clicks, credits_clicks, none_clicks, df_json, language):
    try:
        ctx = callback_context
        if not ctx.triggered:
            return None, ''
        
        # Si no hay datos cargados, usar df_limpio
        if df_json is None:
            from config import df_limpio as df_fallback
            df = df_fallback.copy()
        else:
            # Reconstruir el dataframe desde el JSON
            df = _read_split_json(df_json)
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'botonG':
            from kaplan_meier import plot_km_by_covariate_with_figure
            graph_component, _ = plot_km_by_covariate_with_figure(df, 'gender_F')
            return html.Div([
                html.H3("👥 Kaplan-Meier Curve by Gender" if language == 'en' else "👥 Curva de Kaplan-Meier por Género", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                graph_component
            ]), 'gender_F'
        elif button_id == 'botonDisc':
            from kaplan_meier import plot_km_by_covariate_with_figure
            graph_component, _ = plot_km_by_covariate_with_figure(df, 'disability_N')
            return html.Div([
                html.H3("♿ Kaplan-Meier Curve by Disability" if language == 'en' else "♿ Curva de Kaplan-Meier por Discapacidad", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                graph_component
            ]), 'disability_N'
        elif button_id == 'botonAge':
            from kaplan_meier import plot_km_by_covariate_with_figure
            graph_component, _ = plot_km_by_covariate_with_figure(df, 'age_band')
            return html.Div([
                html.H3("🎂 Kaplan-Meier Curve by Age Band" if language == 'en' else "🎂 Curva de Kaplan-Meier por Banda de Edad", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                graph_component
            ]), 'age_band'
        elif button_id == 'botonEdu':
            from kaplan_meier import plot_km_by_covariate_with_figure
            graph_component, _ = plot_km_by_covariate_with_figure(df, 'highest_education')
            return html.Div([
                html.H3("🎓 Kaplan-Meier Curve by Highest Education" if language == 'en' else "🎓 Curva de Kaplan-Meier por Educación Más Alta", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                graph_component
            ]), 'highest_education'
        elif button_id == 'botonCredits':
            from kaplan_meier import plot_km_by_covariate_with_figure
            graph_component, _ = plot_km_by_covariate_with_figure(df, 'studied_credits')
            return html.Div([
                html.H3("📚 Kaplan-Meier Curve by Studied Credits" if language == 'en' else "📚 Curva de Kaplan-Meier por Créditos Estudiados", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                graph_component
            ]), 'studied_credits'
        elif button_id == 'botonNone':
            return None, ''
        
        # Por si acaso no se detecta ningún botón
        return None, ''
    
    except Exception as e:
        # ✅ Manejo de errores en Kaplan-Meier
        import traceback
        error_div = html.Div([
            html.H3("❌ Error en Kaplan-Meier", style={'color': '#d32f2f', 'fontWeight': 'bold'}),
            html.P(f"No se pudo generar la gráfica: {str(e)}", style={'color': '#1a1a1a'}),
            html.Div(f"Detalles técnicos: {traceback.format_exc()}", style={'fontSize': '12px', 'color': '#666', 'whiteSpace': 'pre-wrap', 'marginTop': '10px'})
        ])
        return error_div, '' 



@app.callback(
    Output('openai-answer-kaplan', 'value'),
    [Input('explicar-btn-kaplan', 'n_clicks')],
    [State('km-current-variable', 'data'),
     State('language-store', 'data'),
     State('df-store', 'data')],
    prevent_initial_call=True 
)
def explicar_kaplan(n_clicks, variable_actual, language, df_json):
    """Callback para generar explicación de la gráfica Kaplan-Meier con datos reales"""
    try:
        if n_clicks is None or n_clicks <= 0:
            return ""
        
        if not variable_actual:
            return f"⚠️  {get_translation(language, 'error_select_variable')}"
        
        if language not in ['es', 'en']:
            language = 'es'
        
        # Extraer estadísticas reales de la variable KM
        stats_info = ""
        if df_json:
            try:
                df = _read_split_json(df_json)
                stats_info = _build_km_interpretation_context(df, variable_actual)
            except Exception as e:
                print(f"Error extrayendo estadísticas KM: {e}")
        
        # Construir prompt con datos reales
        if language == 'en':
            prompt = f"""Write an academic interpretation of the Kaplan-Meier output for '{_humanize_label(variable_actual)}' using only the real group results below:
{stats_info}
Produce exactly 2 short paragraphs: first paragraph for statistical interpretation of curve behavior, second paragraph for practical implication and one limitation. Do not use bullet points."""
        else:
            prompt = f"""Redacta una interpretación académica de la salida Kaplan-Meier para '{_humanize_label(variable_actual)}' usando solo los resultados reales de grupos:
{stats_info}
Devuelve exactamente 2 párrafos breves: el primero con lectura estadística del comportamiento de curvas y el segundo con implicación práctica y una limitación. Sin viñetas ni listas numeradas."""
        
        respuesta = responder_pregunta_con_llama3(prompt, language)
        return respuesta if respuesta else ("⚠️  Error: Empty response" if language == 'en' else "⚠️  Error: Respuesta vacía")
        
    except requests.exceptions.Timeout:
        return f"⚠️  {get_translation(language, 'error_timeout')}"
    except Exception as e:
        print(f"❌ Error en explicar_kaplan: {str(e)}")
        return f"❌ Error: {str(e)}"

# Callback para actualizar el gráfico según la selección del Dropdown
@app.callback(
    [Output('covariables-graph', 'figure'),
    Output('graph-explanation', 'children')],
    [Input('covariables-dropdown', 'value'),
     Input('language-store', 'data')],
    [State('df-store', 'data')],
    prevent_initial_call=False
)
def update_graph(col_chosen, language, df_json):
    # ✅ VALIDACIÓN 1: Leer el dataset desde el store
    df_data = None
    if df_json:
        try:
            df_data = _read_split_json(df_json)
        except:
            df_data = None
    
    # ✅ VALIDACIÓN 2: Verificar que el dataset está cargado y no está vacío
    if df_data is None or df_data.empty:
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=get_translation(language, 'covariate_error_load_dataset'),
            showarrow=False,
            font=dict(size=14, color='red')
        )
        error_fig.update_layout(title=get_translation(language, 'covariate_error_dataset_title'), template="plotly_white")
        return error_fig, html.Div([
            html.H4(get_translation(language, 'covariate_error_data_header'), style={'color': 'red'}),
            html.P(get_translation(language, 'covariate_error_data_body'))
        ])
    
    # ✅ VALIDACIÓN 2: Verificar que language es válido
    if language not in ['es', 'en']:
        language = 'es'

    def _pick_existing(df_local, candidates):
        for col in candidates:
            if col in df_local.columns:
                return col
        return None
    
    try:
        if col_chosen == 'abandono':
            conteo_abandono = df_data['final_result'].value_counts().sort_index()
            fig = go.Figure()
            count_no_abandono = conteo_abandono[0] if 0 in conteo_abandono else 0
            count_abandono = conteo_abandono[1] if 1 in conteo_abandono else 0
            label_no_abandono = get_translation(language, 'no_abandono')
            label_abandono = get_translation(language, 'abandono')
            
            fig.add_trace(go.Bar(x=[label_no_abandono], y=[count_no_abandono], name='', marker_color='#1abc9c'))
            fig.add_trace(go.Bar(x=[label_abandono], y=[count_abandono], name='', marker_color='#006400'))
            fig.update_layout(
                title=get_translation(language, 'abandono_vs_no_abandono'),
                xaxis_title=get_translation(language, 'resultado_final'),
                yaxis_title=get_translation(language, 'num_estudiantes'),
                barmode='group'
            )
            return fig, get_translation(language, 'exp_abandono')
        
        elif col_chosen == 'gender':
            # Soporta datasets con columna original ('gender') o one-hot ('gender_F')
            if 'gender' in df_data.columns:
                plot_df = df_data.copy()
                plot_df['gender_plot'] = plot_df['gender']
            elif 'gender_F' in df_data.columns:
                plot_df = df_data.copy()
                plot_df['gender_plot'] = plot_df['gender_F'].apply(
                    lambda x: get_translation(language, 'femenino') if x == 1 else get_translation(language, 'masculino')
                )
            else:
                raise KeyError('gender / gender_F')

            fig = px.histogram(plot_df, x='gender_plot', color='final_result', barmode='group',
                               title=get_translation(language, 'abandono_genero_title'),
                               color_discrete_map={0: '#1abc9c', 1: '#006400'})
            fig.update_layout(
                xaxis_title=get_translation(language, 'genero'),
                yaxis_title=get_translation(language, 'num_estudiantes'),
                legend_title=get_translation(language, 'abandono')
            )
            return fig, get_translation(language, 'exp_genero')
        
        elif col_chosen == 'disability':
            # Soporta datasets con columna original ('disability') o one-hot ('disability_N')
            if 'disability' in df_data.columns:
                plot_df = df_data.copy()
                plot_df['disability_plot'] = plot_df['disability']
            elif 'disability_N' in df_data.columns:
                plot_df = df_data.copy()
                # disability_N=1 -> sin discapacidad ; disability_N=0 -> con discapacidad
                plot_df['disability_plot'] = plot_df['disability_N'].apply(
                    lambda x: get_translation(language, 'sin_discapacidad') if x == 1 else get_translation(language, 'con_discapacidad')
                )
            else:
                raise KeyError('disability / disability_N')

            fig = px.histogram(plot_df, x='disability_plot', color='final_result', barmode='group',
                               title=get_translation(language, 'abandono_discapacidad_title'),
                               color_discrete_map={0: '#1abc9c', 1: '#006400'})
            fig.update_layout(
                xaxis_title=get_translation(language, 'discapacidad'),
                yaxis_title=get_translation(language, 'num_estudiantes'),
                legend_title=get_translation(language, 'abandono')
            )
            return fig, get_translation(language, 'exp_discapacidad')
        
        elif col_chosen == 'age_band':
            data_age = []
            age_candidates = [
                ('age_band_0-35', '0-35'),
                ('age_band_35-55', '35-55'),
                ('age_band_55<=', '55+'),
                ('age_band_55+', '55+'),
            ]
            available_age = [(col, label) for col, label in age_candidates if col in df_data.columns]
            if not available_age:
                raise KeyError('age_band_*')

            for col, label in available_age:
                for result in [0, 1]:
                    count = len(df_data[(df_data[col] == 1) & (df_data['final_result'] == result)])
                    data_age.append({'age_group': label, 'final_result': result, 'count': count})
            df_plot = pd.DataFrame(data_age)
            fig = px.bar(df_plot, x='age_group', y='count', color='final_result', barmode='group',
                        title=get_translation(language, 'abandono_age_band_title'),
                        color_discrete_map={0: '#1abc9c', 1: '#006400'})
            fig.update_layout(
                xaxis_title=get_translation(language, 'grupo_edad'),
                yaxis_title=get_translation(language, 'num_estudiantes'),
                legend=dict(title=get_translation(language, 'abandono'))
            )
            return fig, get_translation(language, 'exp_age_band')
        
        elif col_chosen == 'highest_education':
            education_candidates = [
                ('highest_education_A Level or Equivalent', 'A Level'),
                ('highest_education_HE Qualification', 'HE Qualification'),
                ('highest_education_Lower Than A Level', 'Lower than A Level'),
                ('highest_education_No Formal quals', 'No Formal'),
                ('highest_education_Post Graduate Qualification', 'Post Graduate'),
            ]
            available_edu = [(col, label) for col, label in education_candidates if col in df_data.columns]
            if not available_edu:
                # Fallback: intentar columna categórica original
                edu_col = _pick_existing(df_data, ['highest_education', 'education'])
                if edu_col is None:
                    raise KeyError('highest_education_* / highest_education')

                plot_df = df_data.copy()
                fig = px.histogram(plot_df, x=edu_col, color='final_result', barmode='group',
                                   title=get_translation(language, 'abandono_highest_education_title'),
                                   color_discrete_map={0: '#1abc9c', 1: '#006400'})
                fig.update_layout(
                    xaxis_title=get_translation(language, 'nivel_educativo'),
                    yaxis_title=get_translation(language, 'num_estudiantes'),
                    legend=dict(title=get_translation(language, 'abandono')),
                    xaxis_tickangle=-45
                )
                return fig, get_translation(language, 'exp_highest_education')

            data_edu = []
            for col, label in available_edu:
                for result in [0, 1]:
                    count = len(df_data[(df_data[col] == 1) & (df_data['final_result'] == result)])
                    data_edu.append({'education': label, 'final_result': result, 'count': count})
            df_plot = pd.DataFrame(data_edu)
            fig = px.bar(df_plot, x='education', y='count', color='final_result', barmode='group',
                        title=get_translation(language, 'abandono_highest_education_title'),
                        color_discrete_map={0: '#1abc9c', 1: '#006400'})
            fig.update_layout(
                xaxis_title=get_translation(language, 'nivel_educativo'),
                yaxis_title=get_translation(language, 'num_estudiantes'),
                legend=dict(title=get_translation(language, 'abandono')),
                xaxis_tickangle=-45
            )
            return fig, get_translation(language, 'exp_highest_education')
        
        elif col_chosen == 'studied_credits':
            if 'studied_credits' not in df_data.columns:
                raise KeyError('studied_credits')
            df_credits = df_data[['studied_credits', 'final_result']].copy()
            
            # ✅ ERROR #6: Validación de quintiles - verificar que hay datos suficientes
            unique_values = df_credits['studied_credits'].nunique()
            if unique_values < 5:
                # Si hay menos de 5 valores únicos, usar menos grupos
                num_groups = max(2, unique_values)
                print(f"⚠️  [STUDIED_CREDITS] Solo {unique_values} valores únicos, usando {num_groups} grupos")
                df_credits['credits_quintile'] = pd.qcut(df_credits['studied_credits'], q=num_groups, duplicates='drop', labels=False)
            else:
                df_credits['credits_quintile'] = pd.qcut(df_credits['studied_credits'], q=5, duplicates='drop', labels=False)
            
            data_credits = []
            for quin in sorted(df_credits['credits_quintile'].unique()):
                for result in [0, 1]:
                    count = len(df_credits[(df_credits['credits_quintile'] == quin) & (df_credits['final_result'] == result)])
                    min_val = df_credits[df_credits['credits_quintile'] == quin]['studied_credits'].min()
                    max_val = df_credits[df_credits['credits_quintile'] == quin]['studied_credits'].max()
                    label = f'Q{int(quin)+1} ({int(min_val)}-{int(max_val)})'
                    data_credits.append({'quintile': label, 'final_result': result, 'count': count})
            df_plot = pd.DataFrame(data_credits)
            fig = px.bar(df_plot, x='quintile', y='count', color='final_result', barmode='group',
                        title=get_translation(language, 'abandono_studied_credits_title'),
                        color_discrete_map={0: '#1abc9c', 1: '#006400'})
            fig.update_layout(
                xaxis_title=get_translation(language, 'creditos_estudiados'),
                yaxis_title=get_translation(language, 'num_estudiantes'),
                legend=dict(title=get_translation(language, 'abandono')),
                height=500
            )
            return fig, get_translation(language, 'exp_studied_credits')
        
        else:
            return go.Figure(), ""
    
    except KeyError as ke:
        # ✅ ERROR 2: Si falta una columna esperada
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"{get_translation(language, 'covariate_error_missing_prefix')} '{str(ke)}'",
            showarrow=False,
            font=dict(size=12, color='red')
        )
        error_fig.update_layout(title=get_translation(language, 'covariate_error_missing_column_title'), template="plotly_white")
        return error_fig, html.Div([
            html.H4(get_translation(language, 'covariate_error_structure_header'), style={'color': 'red'}),
            html.P(f"{get_translation(language, 'covariate_error_missing_prefix')} {str(ke)}")
        ])
    
    except Exception as e:
        # ✅ ERROR 2: Cualquier otro error
        error_fig = go.Figure()
        error_fig.add_annotation(text=f"❌ Error: {str(e)[:40]}...", showarrow=False, font=dict(size=12, color='red'))
        error_fig.update_layout(title=get_translation(language, 'covariate_error_generic_title'), template="plotly_white")
        return error_fig, html.Div([
            html.H4(get_translation(language, 'covariate_error_unexpected_header'), style={'color': 'red'}),
            html.P(str(e))
        ])

@app.callback(
    Output('cox-selected-covariables', 'data'),
    Input('covariables-dropdown-cox', 'value')
)
def update_cox_store(covariables):
    return covariables if covariables else []

@app.callback(
    [Output('cox-regression-output-store', 'data'),
     Output('cox-current-variables', 'data')],
    [Input('cox-selected-covariables', 'data'),
     Input('language-store', 'data'),
     Input('df-store', 'data'),
     Input('dataset-signature-store', 'data')]
)
def update_cox_model(covariables, language, df_json, dataset_signature):
    if covariables is None or len(covariables) == 0:
        return None, ''
    
    if df_json is None:
        return None, ''

    # Reconstruir el dataframe desde el JSON
    df_data = _read_split_json(df_json)
    
    print(f"\n[UPDATE COX] Columnas disponibles en df_data: {list(df_data.columns)}")
    print(f"[UPDATE COX] Shape del dataframe: {df_data.shape}")

    # Asegurarnos de que covariables sea una lista
    if isinstance(covariables, str):  
        covariables = [covariables]
        
    # Llamamos a la función de regresión de Cox con las covariables seleccionadas
    summary, cox_table_html = run_cox_regression(df_data, covariables)
    
    print(f"[UPDATE COX] Summary vacío: {summary.empty}")
    print(f"[UPDATE COX] cox_table_html: {type(cox_table_html)}")
    
    # Guardar datos en formato JSON-serializable para el Store
    store_data = {
        'summary_json': summary.to_json(orient='split') if not summary.empty else None,
        'cox_table_html': cox_table_html,
        'covariables': covariables,
        'dataset_signature': dataset_signature or ''
    }
    
    # Guardar las variables seleccionadas como string para la explicación
    variables_str = ', '.join(covariables) if isinstance(covariables, list) else covariables
    
    print(f"[UPDATE COX] Guardando en Store: {store_data.keys()}")
    return store_data, variables_str


@app.callback(
    Output('cox-regression-output', 'children'),
    [Input('cox-regression-output-store', 'data'),
     Input('language-store', 'data')]
)
def render_cox_output(store_data, language):
    """Renderiza tabla y gráfica de Cox desde el Store"""
    if store_data is None:
        return html.Div([
            html.Br(),
            html.P(get_translation(language, 'respuesta'), style={'textAlign': 'center', 'color': '#999', 'fontSize': '18px'})
        ])
    
    try:
        output_elements = []
        
        # Recrear tabla desde el summary guardado en el Store
        if store_data.get('summary_json'):
            try:
                summary = _read_split_json(store_data['summary_json'])
                if not summary.empty:
                    cox_table = dash_table.DataTable(
                        id='cox-summary-table',
                        columns=[{"name": col, "id": col} for col in summary.columns],
                        data=summary.to_dict('records'),
                        style_table={
                            'height': '300px',
                            'overflowY': 'auto',  
                            'overflowX': 'auto',  
                            'width': '100%'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'lineHeight': '15px',
                            'backgroundColor': 'transparent'
                        },
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
                    )
                    output_elements.append(cox_table)
                    print(f"[RENDER COX] Tabla creada exitosamente: {summary.shape}")
            except Exception as e:
                print(f"[RENDER COX] Error recreando tabla: {e}")
        
        # Recrear y mostrar gráfica si tenemos summary
        if store_data.get('summary_json'):
            try:
                summary = _read_split_json(store_data['summary_json'])
                if not summary.empty:
                    forest_plot = plot_cox_hazard_ratios(
                        summary,
                        'Selected covariates' if language == 'en' else 'Covariables seleccionadas',
                        language=language
                    )
                    output_elements.append(forest_plot)
                    print(f"[RENDER COX] Forest plot creado exitosamente")
            except Exception as e:
                print(f"[RENDER COX] Error recreando forest plot: {e}")
                import traceback
                traceback.print_exc()
        
        if not output_elements:
            return html.Div("No results to display" if language == 'en' else "No hay resultados para mostrar")
        
        print(f"[RENDER COX] Renderizando {len(output_elements)} elementos")
        return html.Div(output_elements, style={'textAlign': 'center', 'marginTop': '20px'})
    
    except Exception as e:
        print(f"[RENDER COX] Error: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(
            f"Error rendering results: {str(e)}" if language == 'en' else f"Error renderizando resultados: {str(e)}"
        )


@app.callback(
    Output('weibull-analysis-output', 'children'),
    [Input('language-store', 'data'), Input('df-store', 'data'), Input('url', 'pathname')]
)
def render_weibull_output(language, df_json, pathname):
    if pathname != '/survival-analysis/weibull':
        raise PreventUpdate

    def _no_data_message():
        return html.Div(
            get_translation(language, 'weibull_no_data'),
            style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
        )

    if df_json is None:
        return _no_data_message()

    try:
        df_data = _read_split_json(df_json)
    except Exception:
        return _no_data_message()

    analysis = build_weibull_analysis(df_data, language=language)
    if not analysis:
        return _no_data_message()

    summary_df = analysis['summary_df']
    table = dash_table.DataTable(
        id='weibull-summary-table',
        columns=[
            {"name": get_translation(language, 'weibull_metric'), "id": "Metrica"},
            {"name": get_translation(language, 'weibull_value'), "id": "Valor"},
        ],
        data=summary_df.to_dict('records'),
        style_table={'overflowX': 'auto', 'maxHeight': '360px', 'overflowY': 'auto', 'marginTop': '10px'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
    )

    return html.Div([
        html.Div([
            html.H3(get_translation(language, 'weibull_summary_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            table,
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'marginBottom': '30px'
        }),
        html.Div([
            html.H3(get_translation(language, 'weibull_graph_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
            dcc.Graph(figure=analysis['figure'], config={'responsive': True, 'displayModeBar': True})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'marginBottom': '30px'
        }),
    ])


@app.callback(
    Output('exponential-analysis-output', 'children'),
    [Input('language-store', 'data'), Input('df-store', 'data'), Input('url', 'pathname')]
)
def render_exponential_output(language, df_json, pathname):
    if pathname != '/survival-analysis/exponential':
        raise PreventUpdate

    def _no_data_message():
        return html.Div(
            get_translation(language, 'exponential_no_data'),
            style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
        )

    if df_json is None:
        return _no_data_message()

    try:
        df_data = _read_split_json(df_json)
    except Exception:
        return _no_data_message()

    analysis = build_exponential_analysis(df_data, language=language)
    if not analysis:
        return _no_data_message()

    summary_df = analysis['summary_df']
    table = dash_table.DataTable(
        id='exponential-summary-table',
        columns=[
            {"name": get_translation(language, 'exponential_metric'), "id": "Metrica"},
            {"name": get_translation(language, 'exponential_value'), "id": "Valor"},
            {"name": get_translation(language, 'exponential_interpretation'), "id": "Interpretacion"},
        ],
        data=summary_df.to_dict('records'),
        style_table={'overflowX': 'auto', 'maxHeight': '360px', 'overflowY': 'auto', 'marginTop': '10px'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
    )

    return html.Div([
        html.Div([
            html.H3(get_translation(language, 'exponential_summary_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            table,
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'marginBottom': '30px'
        }),
        html.Div([
            html.H3(get_translation(language, 'exponential_graph_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
            dcc.Graph(figure=analysis['figure'], config={'responsive': True, 'displayModeBar': True})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'marginBottom': '30px'
        }),
    ])


@app.callback(
    [Output('rsf-analysis-output', 'children'),
     Output('rsf-analysis-data', 'data')],
    [Input('language-store', 'data'), Input('df-store', 'data'), Input('url', 'pathname')]
)
def render_rsf_output(language, df_json, pathname):
    if pathname != '/survival-analysis/rsf':
        raise PreventUpdate

    def _no_data_message():
        return html.Div(
            get_translation(language, 'rsf_no_data'),
            style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
        ), None

    if df_json is None:
        return _no_data_message()

    try:
        df_data = _read_split_json(df_json)
    except Exception:
        return _no_data_message()

    analysis = build_rsf_analysis(df_data, language=language)
    if not analysis:
        return _no_data_message()

    summary_df = analysis['summary_df']
    table = dash_table.DataTable(
        id='rsf-summary-table',
        columns=[
            {"name": get_translation(language, 'rsf_metric'), "id": "Metrica"},
            {"name": get_translation(language, 'rsf_value'), "id": "Valor"},
            {"name": get_translation(language, 'rsf_interpretation'), "id": "Interpretacion"},
        ],
        data=summary_df.to_dict('records'),
        style_table={'overflowX': 'auto', 'maxHeight': '360px', 'overflowY': 'auto', 'marginTop': '10px'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
    )

    output_children = html.Div([
        html.Div([
            html.H3(get_translation(language, 'rsf_summary_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            table,
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'marginBottom': '30px'
        }),
        html.Div([
            html.H3(get_translation(language, 'rsf_graph_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
            dcc.Graph(figure=analysis['figure'], config={'responsive': True, 'displayModeBar': True, 'scrollZoom': False, 'doubleClick': 'reset'})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'marginBottom': '30px'
        }),
        html.Div([
            html.H3(get_translation(language, 'rsf_importance_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
            dcc.Graph(figure=analysis['importance_figure'], config={'responsive': True, 'displayModeBar': True})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'marginBottom': '30px'
        }),
    ])

    store_data = {
        'summary_json': summary_df.to_json(orient='split'),
        'top_feature': analysis['top_feature'],
        'top_feature_importance': analysis['top_feature_importance'],
        'train_c_index': analysis['train_c_index'],
        'oob_score': analysis['oob_score'],
        'n_observations': analysis['n_observations'],
        'n_events': analysis['n_events'],
        'n_features': analysis['n_features'],
        'interpretation': analysis['interpretation'],
        'language': language,
    }

    return output_children, store_data


@app.callback(
    Output('rsf-profile-output', 'children'),
    [Input('rsf-profile-simulate-btn', 'n_clicks'),
     Input('df-store', 'data')],
    [State('rsf-profile-gender', 'value'),
     State('rsf-profile-disability', 'value'),
     State('rsf-profile-age-band', 'value'),
     State('rsf-profile-education', 'value'),
     State('rsf-profile-credits', 'value'),
     State('language-store', 'data')],
    prevent_initial_call=False
)
def simulate_rsf_profile(n_clicks, df_json, gender, disability, age_band, education, credits_level, language):
    if not df_json:
        return html.Div(
            get_translation(language, 'rsf_no_data'),
            style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
        )

    try:
        df_data = _read_split_json(df_json)
    except Exception:
        return html.Div(
            get_translation(language, 'rsf_no_data'),
            style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
        )

    credits_map = {'few': 30, 'medium': 60, 'many': 120}
    profile = {
        'gender_F': gender,
        'disability_N': disability,
        'age_band': age_band,
        'highest_education': education,
        'studied_credits': credits_map.get(credits_level, 30),
    }

    analysis = build_rsf_profile_analysis(df_data, profile, language=language)
    if not analysis:
        return html.Div(
            get_translation(language, 'rsf_no_data'),
            style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
        )

    show_table = bool(n_clicks and n_clicks > 0)

    profile_table = dash_table.DataTable(
        columns=[{"name": "Métrica" if language == 'es' else "Metric", "id": "Metrica"}, {"name": "Valor" if language == 'es' else "Value", "id": "Valor"}],
        data=analysis['summary_df'].to_dict('records'),
        style_table={'overflowX': 'auto', 'maxHeight': '320px', 'overflowY': 'auto', 'marginTop': '10px'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
    )

    profile_title = 'Curva del perfil simulado' if language == 'es' else 'Simulated profile curve'
    profile_summary_title = 'Perfil simulado' if language == 'es' else 'Simulated profile'

    graph_block = html.Div([
        html.H4(profile_title, style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
        dcc.Graph(figure=analysis['figure'], config={'responsive': True, 'displayModeBar': True, 'scrollZoom': False, 'doubleClick': 'reset'})
    ], style={
        'backgroundColor': '#f8f9fa',
        'padding': '18px',
        'borderRadius': '10px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
        'marginBottom': '22px'
    })

    if not show_table:
        return html.Div([
            graph_block,
            html.Div(analysis['interpretation'], style={
                'backgroundColor': '#eef7ff',
                'padding': '14px 16px',
                'borderRadius': '8px',
                'border': '1px solid #d9e8ff',
                'color': '#2c3e50',
                'lineHeight': '1.6'
            })
        ])

    return html.Div([
        html.Div([
            html.H4(profile_summary_title, style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            profile_table,
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '18px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'marginBottom': '22px'
        }),
        graph_block,
        html.Div(analysis['interpretation'], style={
            'backgroundColor': '#eef7ff',
            'padding': '14px 16px',
            'borderRadius': '8px',
            'border': '1px solid #d9e8ff',
            'color': '#2c3e50',
            'lineHeight': '1.6'
        })
    ])


@app.callback(
    Output('openai-answer-rsf', 'value'),
    [Input('btn-rsf', 'n_clicks')],
    [State('rsf-analysis-data', 'data'), State('language-store', 'data')],
    prevent_initial_call=True
)
def explicar_rsf(n_clicks, rsf_store_data, language):
    if n_clicks is None or n_clicks <= 0:
        return get_translation(language, 'respuesta')

    if not rsf_store_data:
        return get_translation(language, 'rsf_no_data')

    top_feature = rsf_store_data.get('top_feature', 'N/A')
    oob_score = rsf_store_data.get('oob_score', None)
    train_c_index = rsf_store_data.get('train_c_index', None)
    n_features = rsf_store_data.get('n_features', 0)

    summary_text = ""
    summary_json = rsf_store_data.get('summary_json')
    if summary_json:
        try:
            summary_df = _read_split_json(summary_json)
            if summary_df is not None and not summary_df.empty:
                summary_lines = []
                for _, row in summary_df.head(12).iterrows():
                    metric = row.get('Metrica', row.get('Metric', 'Métrica'))
                    value = row.get('Valor', row.get('Value', 'N/A'))
                    summary_lines.append(f"- {metric}: {value}")
                summary_text = "\n".join(summary_lines)
        except Exception:
            summary_text = ""

    if language == 'en':
        prompt = (
            "Interpret the RSF dashboard outputs using ONLY the visible chart/table information.\n"
            f"- Predictors: {n_features}\n"
            f"- Train c-index: {train_c_index}\n"
            f"- OOB c-index: {oob_score}\n"
            f"- Top feature: {top_feature}\n"
            f"- Summary table:\n{summary_text}\n"
            "Focus on: 1) calibration/discrimination quality, 2) what the survival curves imply for low/medium/high risk profiles, "
            "3) practical implication and one limitation. Keep it concise and factual."
        )
    else:
        prompt = (
            "Interpreta la salida de RSF usando SOLO la información visible en gráfica y tabla del dashboard.\n"
            f"- Predictores: {n_features}\n"
            f"- c-index entrenamiento: {train_c_index}\n"
            f"- c-index OOB: {oob_score}\n"
            f"- Variable más importante: {top_feature}\n"
            f"- Tabla resumen:\n{summary_text}\n"
            "Enfócate en: 1) calidad de discriminación/calibración, 2) qué implican las curvas para perfiles de riesgo bajo/medio/alto, "
            "3) implicación práctica y una limitación. Sé conciso y factual."
        )

    respuesta = responder_pregunta_con_llama3(prompt, language)
    if respuesta:
        return respuesta

    if language == 'en':
        return (
            f"RSF uses {n_features} predictors. Train c-index={train_c_index}, OOB c-index={oob_score}, and top feature={top_feature}. "
            "The dashboard curves suggest meaningful separation across risk profiles; confirm with external validation."
        )
    return (
        f"RSF usa {n_features} predictores. c-index entrenamiento={train_c_index}, c-index OOB={oob_score}, y variable principal={top_feature}. "
        "Las curvas del dashboard sugieren separación entre perfiles de riesgo; conviene validar externamente."
    )


@app.callback(
    [Output('openai-answer-weibull', 'children'),
     Output('weibull-ai-text-store', 'data'),
     Output('weibull-ai-language-store', 'data')],
    [Input('btn-weibull', 'n_clicks')],
    [State('df-store', 'data'), State('language-store', 'data')],
    prevent_initial_call=True
)
def explicar_weibull(n_clicks, df_json, language):
    if n_clicks is None or n_clicks <= 0:
        return "", "", ""

    if not df_json:
        no_data_msg = get_translation(language, 'weibull_no_data')
        return no_data_msg, "", ""

    try:
        df_data = _read_split_json(df_json)
        analysis = build_weibull_analysis(df_data, language=language)
        if not analysis:
            no_data_msg = get_translation(language, 'weibull_no_data')
            return no_data_msg, "", ""

        def _fallback_weibull_explanation():
            shape = analysis['shape']
            scale = analysis['scale']
            median = analysis['median_survival']
            event_rate = analysis['event_rate']
            risk_text = (
                'aumenta con el tiempo' if shape > 1.05 else
                'disminuye con el tiempo' if shape < 0.95 else
                'se mantiene aproximadamente constante'
            )

            if language == 'en':
                return (
                    f"The Weibull model fitted to the cleaned dataset gives a shape parameter of {shape:.3f}, which indicates that the risk {('increases' if shape > 1.05 else 'decreases' if shape < 0.95 else 'remains approximately constant')} over time. "
                    f"The scale parameter is {scale:.3f} and the median survival time is {median:.3f}. The event rate is {event_rate:.1f}%, so the fitted curve summarizes the overall survival pattern of the cleaned data."
                )

            return (
                f"El modelo Weibull ajustado al dataset limpio da un parámetro de forma de {shape:.3f}, lo que significa que el riesgo {risk_text}. "
                f"El parámetro de escala es {scale:.3f} y la mediana de supervivencia es {median:.3f}. La tasa de eventos es {event_rate:.1f}%, por lo que la curva ajustada resume el patrón global de supervivencia de los datos limpios."
            )

        respuesta = generate_interpretation_for_pdf(
            'weibull',
            {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'event_rate': analysis['event_rate'],
                'variable_name': 'Weibull'
            },
            analysis['summary_df'],
            language=language
        )
        final_text = respuesta if respuesta else _fallback_weibull_explanation()
        return final_text, final_text, language
    except Exception as e:
        print(f"❌ Error en explicar_weibull: {str(e)}")
        error_msg = get_translation(language, 'weibull_no_data')
        return error_msg, "", ""


@app.callback(
    [Output('openai-answer-exponential', 'children'),
     Output('exponential-ai-text-store', 'data'),
     Output('exponential-ai-language-store', 'data')],
    [Input('btn-exponential', 'n_clicks')],
    [State('df-store', 'data'), State('language-store', 'data')],
    prevent_initial_call=True
)
def explicar_exponential(n_clicks, df_json, language):
    if n_clicks is None or n_clicks <= 0:
        return "", "", ""

    if not df_json:
        no_data_msg = get_translation(language, 'exponential_no_data')
        return no_data_msg, "", ""

    try:
        df_data = _read_split_json(df_json)
        analysis = build_exponential_analysis(df_data, language=language)
        if not analysis:
            no_data_msg = get_translation(language, 'exponential_no_data')
            return no_data_msg, "", ""

        respuesta = generate_interpretation_for_pdf(
            'exponential',
            {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'event_rate': analysis['event_rate'],
                'variable_name': 'Exponential'
            },
            analysis['summary_df'],
            language=language
        )
        final_text = respuesta if respuesta else analysis['interpretation']
        return final_text, final_text, language
    except Exception as e:
        print(f"❌ Error en explicar_exponential: {str(e)}")
        error_msg = get_translation(language, 'exponential_no_data')
        return error_msg, "", ""

@app.callback(
    Output('openai-answer-cox', 'value'),
    [Input('btn-cox', 'n_clicks')],
    [State('cox-regression-output-store', 'data'),
     State('cox-current-variables', 'data'),
     State('language-store', 'data'),
     State('dataset-signature-store', 'data')],
    prevent_initial_call=True
)
def explicar_cox(n_clicks, cox_store_data, variables_seleccionadas, language, dataset_signature):
    """Callback para generar explicación de Cox Regression con datos reales"""
    try:
        if n_clicks is None or n_clicks <= 0:
            return ""
        
        if not variables_seleccionadas:
            return f"⚠️  {get_translation(language, 'error_select_covariate')}"

        if not cox_store_data or not isinstance(cox_store_data, dict) or not cox_store_data.get('summary_json'):
            return (
                "⚠️  Primero ejecuta la regresión de Cox seleccionando covariables y esperando a que aparezca la tabla."
                if language == 'es' else
                "⚠️  Run Cox regression first: select covariates and wait until the summary table appears."
            )

        if cox_store_data.get('dataset_signature', '') != (dataset_signature or ''):
            return (
                "⚠️  El dataset ha cambiado. Recalcula la regresión de Cox antes de generar la explicación."
                if language == 'es' else
                "⚠️  The dataset changed. Recalculate Cox regression before generating the explanation."
            )
        
        if language not in ['es', 'en']:
            language = 'es'
        
        # Extraer datos de la tabla Cox si están disponibles
        table_summary = ""
        if cox_store_data and isinstance(cox_store_data, dict):
            try:
                if 'summary_json' in cox_store_data and cox_store_data['summary_json']:
                    summary_df = _read_split_json(cox_store_data['summary_json'])
                    table_summary = "\nDatos de la tabla Cox:"
                    for _, row in summary_df.iterrows():
                        cov_name = _humanize_label(row.get('Covariable', 'N/A'))
                        hr = row.get('exp(Coef.)', row.get('exp(coef)', 'N/A'))
                        p_val = row.get('p', 'N/A')
                        hr_text = f"{float(hr):.3f}" if hr not in [None, 'N/A'] else 'N/A'
                        p_text = f"{float(p_val):.4f}" if p_val not in [None, 'N/A'] else 'N/A'
                        table_summary += f"\n- {cov_name}: HR={hr_text}, p={p_text}"
            except Exception as e:
                print(f"Error extrayendo tabla Cox: {e}")
        
        # Construir prompt con datos reales
        if language == 'en':
            prompt = (f"Write an academic interpretation of Cox Regression for {_humanize_label(variables_seleccionadas)} using only this real table summary:{table_summary}\n"
                     f"Return 2 short paragraphs: first paragraph with significant/non-significant covariates and HR interpretation (risk vs protective), "
                     f"second paragraph with practical conclusion and one methodological limitation. No bullet points.")
        else:
            prompt = (f"Redacta una interpretación académica de la Regresión de Cox para {_humanize_label(variables_seleccionadas)} usando solo este resumen real de tabla:{table_summary}\n"
                     f"Devuelve 2 párrafos breves: el primero con covariables significativas/no significativas y lectura de HR (riesgo vs protector), "
                     f"el segundo con conclusión práctica y una limitación metodológica. Sin viñetas ni listas.")
        
        respuesta = responder_pregunta_con_llama3(prompt, language)
        return respuesta if respuesta else ("⚠️  Error: Empty response" if language == 'en' else "⚠️  Error: Respuesta vacía")
        
    except requests.exceptions.Timeout:
        return f"⚠️  {get_translation(language, 'error_timeout')}"
    except Exception as e:
        print(f"❌ Error en explicar_cox: {str(e)}")
        return f"❌ Error: {str(e)}"
        return f"❌ Error al generar explicación: {str(e)}" 

#callback para el control del analisis de log rank
@app.callback(
    Output('logrank-selected-covariables', 'data'),
    Input('covariables-dropdown-logrank', 'value')
)
def update_logrank_store(covariables):
    return covariables if covariables else []

@app.callback(
    [Output('logrank-test-output-store', 'data'),
     Output('logrank-current-variable', 'data')],
    [Input('logrank-selected-covariables', 'data'),
     Input('language-store', 'data'),
     Input('df-store', 'data'),
     Input('dataset-signature-store', 'data')]
)

def update_logrank_test(covariables, language, df_json, dataset_signature):
    # Verificar que al menos se haya seleccionado una covariable
    if not covariables:
        return None, ''
    
    if df_json is None:
        return None, ''

    # Reconstruir el dataframe desde el JSON
    df_data = _read_split_json(df_json)
    
    print(f"\n[UPDATE LOGRANK] Covariables seleccionadas: {covariables}")
    print(f"[UPDATE LOGRANK] Shape del dataframe: {df_data.shape}")

    panels = []
    results_payload = []

    # Ejecutar el Test de Log-Rank para cada covariable seleccionada
    for covariable in covariables:
        print(f"\n[UPDATE LOGRANK] Procesando covariable: {covariable}")
        try:
            res_df = perform_log_rank_test(df_data, covariable)
            print(f"[UPDATE LOGRANK] Resultado para {covariable}: {len(res_df)} filas")
            if res_df.empty:
                print(f"[WARNING LOGRANK] res_df vacío para {covariable}")
                continue
            
            # Crear panel con tabla de resultados del test y gráfica Kaplan-Meier estratificada
            table = display_logrank_summary_table(res_df)
            graph_component = plot_logrank_curves(df_data, covariable, language=language)

            results_payload.append({
                'covariable': covariable,
                'results_json': res_df.to_json(orient='split')
            })
            
            panels.append(html.Div([
                html.H3(f"Resultado del Test de Log-Rank para {covariable}", style={'textAlign': 'center'}),
                graph_component,
                table
            ]))
            

                
        except Exception as e:
            print(f"[ERROR LOGRANK] Error procesando {covariable}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Guardar datos estructurados para la explicación y para renderizar la vista
    variables_str = ', '.join(covariables) if isinstance(covariables, list) else covariables
    store_data = {
        'df_json': df_json,
        'covariables': covariables,
        'results': results_payload,
        'dataset_signature': dataset_signature or ''
    }

    return store_data, variables_str

@app.callback(
    Output('openai-answer-logrank', 'value'),
    [Input('explicar-btn-logrank', 'n_clicks')],
    [State('logrank-test-output-store', 'data'),
     State('logrank-current-variable', 'data'),
    State('language-store', 'data'),
    State('dataset-signature-store', 'data')]  
)
def explicar_logrank(n_clicks, logrank_content, variables_seleccionadas, language, dataset_signature):
    """Callback para generar explicación de Log-Rank Test con datos reales"""
    try:
        if n_clicks is None or n_clicks <= 0:
            return ""
        
        if not logrank_content:
            return f"⚠️  {get_translation(language, 'error_select_logrank')}"

        if isinstance(logrank_content, dict) and logrank_content.get('dataset_signature', '') != (dataset_signature or ''):
            return (
                "⚠️  El dataset ha cambiado. Recalcula el Test de Log-Rank antes de generar la explicación."
                if language == 'es' else
                "⚠️  The dataset changed. Recalculate the Log-Rank test before generating the explanation."
            )
        
        if language not in ['es', 'en']:
            language = 'es'
        
        # Formatear datos del store (limitar para prompt conciso)
        logrank_data_str = _build_logrank_interpretation_context(logrank_content)
        
        # Construir prompt con datos reales
        if language == 'en':
            prompt = (f"Write an academic interpretation of the Log-Rank output for {_humanize_label(variables_seleccionadas)} using only the real comparisons below:\n"
                     f"{logrank_data_str}\n"
                     f"Return 2 short paragraphs in prose: first paragraph identifying statistically significant contrasts and direction of differences, "
                     f"second paragraph with practical reading of survival-curve separation and one limitation. No bullet points.")
        else:
            prompt = (f"Redacta una interpretación académica del Test Log-Rank para {_humanize_label(variables_seleccionadas)} usando solo las comparaciones reales:\n"
                     f"{logrank_data_str}\n"
                     f"Devuelve 2 párrafos en prosa: el primero identificando contrastes significativos y diferencias principales, "
                     f"el segundo con lectura práctica de la separación de curvas y una limitación. Sin viñetas ni listas numeradas.")
        
        respuesta = responder_pregunta_con_llama3(prompt, language)
        if len(respuesta) > 3000:
            max_length = 1500
            chunks = [respuesta[i:i + max_length] for i in range(0, len(respuesta), max_length)]
            return '\n\n'.join(chunks)
        return respuesta if respuesta else ("⚠️  Error: Empty response" if language == 'en' else "⚠️  Error: Respuesta vacía")
        
    except requests.exceptions.Timeout:
        return f"⚠️  {get_translation(language, 'error_timeout')}"
    except Exception as e:
        print(f"❌ Error en explicar_logrank: {str(e)}")
        return f"❌ Error: {str(e)}"

# Callbacks para sincronizar stores con divs en las páginas
@app.callback(
    Output('logrank-test-output', 'children'),
    [Input('logrank-test-output-store', 'data'),
     Input('language-store', 'data')]
)
def sync_logrank_output(data, language):
    if not data or not isinstance(data, dict):
        return data

    try:
        df_json = data.get('df_json')
        covariables = data.get('covariables') or []
        results = data.get('results') or []

        if df_json:
            df_data = _read_split_json(df_json)
        else:
            df_data = None

        panels = []
        for result_info in results:
            covariable = result_info.get('covariable')
            results_json = result_info.get('results_json')
            if not covariable or not results_json or df_data is None:
                continue
            res_df = _read_split_json(results_json)
            if res_df.empty:
                continue
            table = display_logrank_summary_table(res_df)
            graph_component = plot_logrank_curves(df_data, covariable, language=language)
            panels.append(html.Div([
                html.H3(
                    f"Log-Rank Test results for {covariable}" if language == 'en' else f"Resultado del Test de Log-Rank para {covariable}",
                    style={'textAlign': 'center'}
                ),
                graph_component,
                table
            ]))

        return html.Div(panels) if panels else html.Div("No results to display" if language == 'en' else "No hay resultados para mostrar")
    except Exception as e:
        print(f"[SYNC LOGRANK] Error reconstruyendo salida: {e}")
        return html.Div(
            f"Error rendering results: {str(e)}" if language == 'en' else f"Error renderizando resultados: {str(e)}"
        )


if __name__ == '__main__':
    register_pdf_export_callbacks(app)  # PDF export HABILITADO
    app.run_server(debug=True, port=8050)
