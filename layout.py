import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output  
import plotly.express as px
import base64
import io
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc 
import numpy as np
from translations import get_translation
from pdf_exporter import export_survival_analysis_to_pdf

df = pd.read_csv(r'C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis\dataset_limpio.csv', sep=';')
df['abandono'] = df['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)

df['gender'] = df['gender_F'].map({1: 'Femenino', 0: 'Masculino'})

df['disability'] = df['disability_N'].map({1: 'Con Discapacidad', 0: 'Sin Discapacidad'})

# Función para mapear age_band desde one-hot encoding
def map_age_band(row):
    if row.get('age_band_0-35', 0) == 1:
        return '0-35 años'
    elif row.get('age_band_35-55', 0) == 1:
        return '35-55 años'
    elif row.get('age_band_55<=', 0) == 1:
        return '55+ años'
    return 'Desconocido'

# Función para mapear highest_education desde one-hot encoding
def map_highest_education(row):
    education_map = {
        'highest_education_A Level or Equivalent': 'A Level o Equivalente',
        'highest_education_HE Qualification': 'Calificación HE',
        'highest_education_Lower Than A Level': 'Inferior a A Level',
        'highest_education_No Formal quals': 'Sin Cualificaciones',
        'highest_education_Post Graduate Qualification': 'Postgrado'
    }
    for col, label in education_map.items():
        if col in row.index and row[col] == 1:
            return label
    return 'Desconocido'

# Crear columnas derivadas para los nuevos atributos (solo si existen)
if all(col in df.columns for col in ['age_band_0-35', 'age_band_35-55', 'age_band_55<=']):
    df['age_band'] = df.apply(map_age_band, axis=1)

if any(col in df.columns for col in ['highest_education_A Level or Equivalent', 'highest_education_HE Qualification',
                                      'highest_education_Lower Than A Level', 'highest_education_No Formal quals',
                                      'highest_education_Post Graduate Qualification']):
    df['highest_education'] = df.apply(map_highest_education, axis=1)


# ===== COMPONENTE MODAL PARA EXPORTAR A PDF =====
def create_pdf_export_modal(modal_id, analysis_type="kaplan-meier", language='es'):
    """
    Crea un modal para seleccionar qué incluir en la exportación a PDF
    
    Args:
        modal_id: ID único del modal
        analysis_type: tipo de análisis ('kaplan-meier', 'cox-regression', 'log-rank')
    """
    if analysis_type == 'weibull-exponential-combined':
        return html.Div([
            html.Div(
                id=f"{modal_id}-overlay",
                style={
                    'display': 'none',
                    'position': 'fixed',
                    'top': 0,
                    'left': 0,
                    'width': '100%',
                    'height': '100%',
                    'backgroundColor': 'rgba(0,0,0,0.5)',
                    'zIndex': 999,
                    'cursor': 'pointer'
                }
            ),
            html.Div(
                id=f"{modal_id}-container",
                style={
                    'display': 'none',
                    'position': 'fixed',
                    'top': '50%',
                    'left': '50%',
                    'transform': 'translate(-50%, -50%)',
                    'backgroundColor': 'white',
                    'padding': '30px',
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
                    'zIndex': 1000,
                    'width': '90%',
                    'maxWidth': '560px',
                    'maxHeight': '650px',
                    'overflowY': 'auto'
                },
                children=[
                    html.Div([
                        html.H2(
                            "📄 Informe combinado: Weibull + Exponencial" if language == 'es' else "📄 Combined report: Weibull + Exponential",
                            style={'margin': 0, 'color': '#2c3e50'}
                        ),
                        html.Button(
                            "✕",
                            id=f"{modal_id}-close-btn",
                            style={
                                'position': 'absolute',
                                'top': '15px',
                                'right': '15px',
                                'background': 'none',
                                'border': 'none',
                                'fontSize': '24px',
                                'cursor': 'pointer',
                                'color': '#999'
                            }
                        )
                    ], style={'position': 'relative', 'marginBottom': '20px'}),

                    html.Div([
                        html.Label(
                            "📁 Nombre del informe" if language == 'es' else "📁 Report name",
                            style={'fontWeight': 'bold', 'color': '#34495e'}
                        ),
                        dcc.Input(
                            id=f"{modal_id}-filename",
                            type="text",
                            placeholder=("Ejemplo: informe_weibull_exponencial" if language == 'es' else "Example: weibull_exponential_report"),
                            style={
                                'width': '100%',
                                'padding': '8px',
                                'border': '1px solid #ddd',
                                'borderRadius': '6px',
                                'marginTop': '5px',
                                'boxSizing': 'border-box'
                            }
                        ),
                        html.Small(
                            "No incluir .pdf" if language == 'es' else "Do not include .pdf",
                            style={'display': 'block', 'color': '#999', 'marginTop': '5px'}
                        )
                    ], style={'marginBottom': '18px'}),

                    html.Div([
                        html.H4("🧪 Técnicas" if language == 'es' else "🧪 Techniques", style={'color': '#34495e', 'marginTop': 0}),
                        dcc.Checklist(
                            id=f"{modal_id}-techniques",
                            options=[
                                {'label': ' Weibull', 'value': 'weibull'},
                                {'label': ' Exponencial' if language == 'es' else ' Exponential', 'value': 'exponential'}
                            ],
                            value=['weibull', 'exponential'],
                            style={'marginBottom': '8px', 'lineHeight': '1.8'},
                            labelStyle={'display': 'block', 'marginBottom': '8px'}
                        )
                    ], style={'marginBottom': '14px', 'padding': '15px', 'backgroundColor': '#f8fbff', 'borderRadius': '8px'}),

                    html.Div([
                        html.H4("📋 Contenido" if language == 'es' else "📋 Content", style={'color': '#34495e', 'marginTop': 0}),
                        dcc.Checklist(
                            id=f"{modal_id}-content",
                            options=[
                                {'label': ' Resumen' if language == 'es' else ' Summary', 'value': 'summary'},
                                {'label': ' Tabla de resultados' if language == 'es' else ' Results table', 'value': 'table'},
                                {'label': ' Gráfica' if language == 'es' else ' Graph', 'value': 'graph'},
                                {'label': ' Interpretación IA (opcional)' if language == 'es' else ' AI interpretation (optional)', 'value': 'ai_interpretation'}
                            ],
                            value=['summary', 'table', 'graph'],
                            style={'marginBottom': '8px', 'lineHeight': '1.8'},
                            labelStyle={'display': 'block', 'marginBottom': '8px'}
                        )
                    ], style={'marginBottom': '14px', 'padding': '15px', 'backgroundColor': '#f8fbff', 'borderRadius': '8px'}),

                    html.Small(
                        "Nota: Exponencial se incluirá como caso particular del modelo Weibull."
                        if language == 'es' else
                        "Note: Exponential will be included as a special case of the Weibull model.",
                        style={'display': 'block', 'color': '#5d6d7e', 'fontStyle': 'italic', 'marginBottom': '18px'}
                    ),

                    html.Div([
                        html.Button(
                            "Cancelar" if language == 'es' else "Cancel",
                            id=f"{modal_id}-cancel-btn",
                            style={
                                'padding': '10px 20px',
                                'marginRight': '10px',
                                'backgroundColor': '#95a5a6',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Button(
                            "Descargar PDF" if language == 'es' else "Download PDF",
                            id=f"{modal_id}-download-btn",
                            style={
                                'padding': '10px 20px',
                                'backgroundColor': '#3498db',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold',
                                'boxShadow': '0 2px 8px rgba(52, 152, 219, 0.3)'
                            }
                        ),
                    ], style={'textAlign': 'right'}),

                    html.Div(
                        id=f"{modal_id}-error",
                        style={
                            'display': 'none',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }
                    ),

                    dcc.Download(id=f"{modal_id}-download"),
                ]
            )
        ])

    # Todas las opciones comienzan DESMARCADAS por defecto
    default_values = []
    
    # Para KM: crear opciones sin tabla (se oculta con CSS)
    if analysis_type == 'kaplan-meier':
        options = [
            {'label': f" {'Resumen general' if language == 'es' else 'Summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    elif analysis_type == 'rsf':
        options = [
            {'label': f" {get_translation(language, 'rsf_pdf_option_general_summary')}", 'value': 'general_summary'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_model_summary')}", 'value': 'model_summary'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_graph')}", 'value': 'graph'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_importance')}", 'value': 'importance'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_profile')}", 'value': 'profile'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_ai')}", 'value': 'ai_interpretation'},
        ]
    elif analysis_type == 'weibull':
        options = [
            {'label': f" {'Resumen general' if language == 'es' else 'Summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Tabla de resultados' if language == 'es' else 'Results table'}", 'value': 'table'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    elif analysis_type == 'exponential':
        options = [
            {'label': f" {'Resumen del ajuste exponencial' if language == 'es' else 'Exponential fit summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    else:  # cox-regression, log-rank
        options = [
            {'label': f" {'Resumen general' if language == 'es' else 'Summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Tabla de resultados' if language == 'es' else 'Results table'}", 'value': 'table'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    
    return html.Div([
        # Overlay oscuro del modal
        html.Div(
            id=f"{modal_id}-overlay",
            style={
                'display': 'none',
                'position': 'fixed',
                'top': 0,
                'left': 0,
                'width': '100%',
                'height': '100%',
                'backgroundColor': 'rgba(0,0,0,0.5)',
                'zIndex': 999,
                'cursor': 'pointer'
            }
        ),
        
        # Caja del modal
        html.Div(
            id=f"{modal_id}-container",
            style={
                'display': 'none',
                'position': 'fixed',
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'backgroundColor': 'white',
                'padding': '30px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
                'zIndex': 1000,
                'width': '90%',
                'maxWidth': '500px',
                'maxHeight': '600px',
                'overflowY': 'auto'
            },
            children=[
                # Header
                html.Div([
                    html.H2(f"📄 {get_translation(language, 'pdf_modal_title')}", style={'margin': 0, 'color': '#2c3e50'}),
                    html.Button(
                        "✕",
                        id=f"{modal_id}-close-btn",
                        style={
                            'position': 'absolute',
                            'top': '15px',
                            'right': '15px',
                            'background': 'none',
                            'border': 'none',
                            'fontSize': '24px',
                            'cursor': 'pointer',
                            'color': '#999'
                        }
                    )
                ], style={'position': 'relative', 'marginBottom': '20px'}),
                
                # Opciones de exportación
                html.Div([
                    # Instrucciones
                    html.P(
                        get_translation(language, 'pdf_modal_instruction'),
                        style={'color': '#555', 'fontSize': '14px', 'fontStyle': 'italic', 'marginBottom': '15px'}
                    ),
                    
                    # Sección: Contenido
                    html.Div([
                        html.H4(f"📋 {get_translation(language, 'pdf_modal_content')}", style={'color': '#34495e', 'marginTop': 0}),
                        
                        dcc.Checklist(
                            id=f"{modal_id}-checklist-content",
                            options=options,  # Opciones dinámicas según el análisis
                            value=default_values,  # Todas desmarcadas por defecto
                            style={'marginBottom': '15px', 'lineHeight': '1.8'},
                            labelStyle={'display': 'block', 'marginBottom': '8px'}
                        )
                    ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#f8fbff', 'borderRadius': '8px'}),
                    
                    # Nombre del archivo
                    html.Div([
                        html.Label(f"📁 {get_translation(language, 'pdf_modal_filename_label')}", style={'fontWeight': 'bold', 'color': '#34495e'}),
                        dcc.Input(
                            id=f"{modal_id}-filename",
                            type="text",
                            placeholder=get_translation(language, 'pdf_modal_filename_placeholder'),
                            style={
                                'width': '100%',
                                'padding': '8px',
                                'border': '1px solid #ddd',
                                'borderRadius': '6px',
                                'marginTop': '5px',
                                'boxSizing': 'border-box'
                            }
                        ),
                        html.Small(
                            get_translation(language, 'pdf_modal_no_extension'),
                            style={'display': 'block', 'color': '#999', 'marginTop': '5px'}
                        )
                        ], style={'marginBottom': '20px'}),
                    ], style={'marginBottom': '20px'}),

                    # Botones de acción
                    html.Div([
                        html.Button(
                            get_translation(language, 'pdf_modal_cancel'),
                            id=f"{modal_id}-cancel-btn",
                            style={
                                'padding': '10px 20px',
                                'marginRight': '10px',
                                'backgroundColor': '#95a5a6',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Button(
                            get_translation(language, 'pdf_modal_download'),
                            id=f"{modal_id}-download-btn",
                            style={
                                'padding': '10px 20px',
                                'backgroundColor': '#3498db',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold',
                                'boxShadow': '0 2px 8px rgba(52, 152, 219, 0.3)'
                            }
                        ),
                    ], style={'textAlign': 'right'}),

                html.Div(
                    id=f"{modal_id}-error",
                    style={
                        'display': 'none',
                        'marginTop': '15px',
                        'padding': '10px 12px',
                        'borderRadius': '6px',
                        'backgroundColor': '#fff1f0',
                        'color': '#c0392b',
                        'border': '1px solid #f5c6cb',
                        'fontSize': '14px',
                        'fontWeight': 'bold'
                    }
                ),
                
                # Componente de descarga
                dcc.Download(id=f"{modal_id}-download"),
            ]
        ),
        
        # Store para mantener datos del análisis actual
        dcc.Store(id=f"{modal_id}-analysis-data")
    ])


# Funciones para crear páginas dinámicamente con soporte multilingüe
def create_survival_analysis_page(language='es'):
    techniques = [
        {
            'image': '/assets/weibull.svg',
            'image_style': {'width': '78%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '20px'},
            'label_key': 'weibull_analysis',
            'href': '/survival-analysis/weibull'
        },
        {
            'image': '/assets/rsf.svg',
            'image_style': {'width': '78%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '20px'},
            'label_key': 'random_survival_forest',
            'href': '/survival-analysis/rsf'
        },
        {
            'image': '/assets/exponential.svg',
            'image_style': {'width': '78%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '20px'},
            'label_key': 'exponential_analysis',
            'href': '/survival-analysis/exponential'
        },
        {
            'image': '/assets/kaplan.png',
            'image_style': {'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '15px'},
            'label_key': 'kaplan_meier',
            'href': '/survival-analysis/kaplan-meier'
        },
        {
            'image': '/assets/cox.png',
            'image_style': {'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '30px'},
            'label_key': 'cox_regression',
            'href': '/survival-analysis/cox-regression'
        },
        {
            'image': '/assets/logrank.png',
            'image_style': {'width': '50%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '30px'},
            'label_key': 'log_rank_test',
            'href': '/survival-analysis/log-rank'
        },
    ]

    return html.Div([  
        html.H1(get_translation(language, 'survival_analysis'), style={'textAlign': 'center'}),
        # Barra de navegación interna
        html.Div([
            html.Div([
                html.Img(src=technique['image'], style=technique['image_style']),
                dcc.Link(get_translation(language, technique['label_key']), href=technique['href'], className='home-link'),
            ], className='survival-technique-card', style={'textAlign': 'center', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%', 'minWidth': '220px'})
            for technique in techniques
        ], className='survival-techniques-grid', style={'textAlign': 'center', 'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '30px', 'marginTop': '20px', 'paddingBottom': '90px'}),  

        # Contenedor para mostrar el gráfico de Kaplan-Meier
        html.Div(id='survival_analysis_page', children=[]),
    ])

# Páginas predefinidas (se actualizarán dinámicamente con callbacks)
survival_analysis_page = create_survival_analysis_page()

def create_weibull_analysis_page(language='es'):
    return html.Div([
        html.H1(
            get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'weibull_analysis')),
            style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '20px', 'color': '#1a1a1a', 'fontWeight': 'bold'}
        ),

        html.Div([
            html.P(
                get_translation(language, 'weibull_intro'),
                style={'margin': 0, 'fontSize': '1.05em', 'lineHeight': '1.6', 'color': '#34495e'}
            ),
            html.P(
                get_translation(language, 'weibull_note'),
                style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic'}
            ),
        ], style={
            'backgroundColor': '#f8fbff',
            'padding': '22px',
            'borderRadius': '10px',
            'border': '1px solid #dfe9f3',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'margin': '0 20px 25px 20px'
        }),

        html.Div(
            dcc.Loading(
                id='loading-weibull',
                type='circle',
                children=html.Div(id='weibull-analysis-output')
            ),
            style={'margin': '0 20px 30px 20px'}
        ),

        html.Div([
            html.Button(get_translation(language, 'explicar_weibull'), id='btn-weibull',
                       title=("Primero carga y preprocesa un dataset" if language == 'es' else "Load and preprocess a dataset first"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"📄 {'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-weibull-btn',
                       title=("Necesitas resultados Weibull válidos para exportar" if language == 'es' else "You need valid Weibull results before exporting"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(
                "📄 Exportar informe combinado" if language == 'es' else "📄 Export combined report",
                id='export-weibexp-btn',
                title=("Requiere análisis Weibull y Exponencial calculados" if language == 'es' else "Requires computed Weibull and Exponential analyses"),
                style={'padding': '10px 20px', 'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none',
                       'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}
            )
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

        html.Div(
            id='openai-answer-weibull',
            children=get_translation(language, 'respuesta'),
            style={
                'width': '80%',
                'minHeight': '260px',
                'whiteSpace': 'pre-wrap',
                'margin': '20px auto',
                'display': 'block',
                'border': '1px solid #ccc',
                'borderRadius': '8px',
                'fontSize': '14px',
                'padding': '14px',
                'overflowY': 'auto',
                'color': '#1a1a1a',
                'backgroundColor': '#fff'
            }
        ),

        create_pdf_export_modal('weibull-pdf-modal', 'weibull', language),
        create_pdf_export_modal('weibexp-pdf-modal', 'weibull-exponential-combined', language)
    ])

weibull_analysis_page = create_weibull_analysis_page()


def create_exponential_analysis_page(language='es'):
    return html.Div([
        html.H1(
            get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'exponential_analysis')),
            style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '20px', 'color': '#1a1a1a', 'fontWeight': 'bold'}
        ),

        html.Div([
            html.P(
                get_translation(language, 'exponential_intro'),
                style={'margin': 0, 'fontSize': '1.05em', 'lineHeight': '1.6', 'color': '#34495e'}
            ),
            html.P(
                get_translation(language, 'exponential_note'),
                style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic'}
            ),
        ], style={
            'backgroundColor': '#f8fbff',
            'padding': '22px',
            'borderRadius': '10px',
            'border': '1px solid #dfe9f3',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'margin': '0 20px 25px 20px'
        }),

        html.Div(
            dcc.Loading(
                id='loading-exponential',
                type='circle',
                children=html.Div(id='exponential-analysis-output')
            ),
            style={'margin': '0 20px 30px 20px'}
        ),

          html.Div([
            html.Button(get_translation(language, 'explicar_exponential'), id='btn-exponential',
                                         title=("Primero carga y preprocesa un dataset" if language == 'es' else "Load and preprocess a dataset first"),
                     style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                         'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"📄 {'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-exponential-btn',
                                         title=("Necesitas resultados Exponencial válidos para exportar" if language == 'es' else "You need valid Exponential results before exporting"),
                     style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                                 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
                        html.Button(
                                "📄 Exportar informe combinado" if language == 'es' else "📄 Export combined report",
                                id='export-weibexp-btn',
                                                                title=("Requiere análisis Weibull y Exponencial calculados" if language == 'es' else "Requires computed Weibull and Exponential analyses"),
                                style={'padding': '10px 20px', 'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none',
                                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}
                        ),
          ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

        html.Div(
            id='openai-answer-exponential',
            children=get_translation(language, 'respuesta'),
            style={
                'width': '80%',
                'minHeight': '220px',
                'whiteSpace': 'pre-wrap',
                'margin': '20px auto',
                'display': 'block',
                'border': '1px solid #ccc',
                'borderRadius': '8px',
                'fontSize': '14px',
                'padding': '14px',
                'overflowY': 'auto',
                'color': '#1a1a1a',
                'backgroundColor': '#fff'
            }
        ),

        create_pdf_export_modal('exponential-pdf-modal', 'exponential', language),
        create_pdf_export_modal('weibexp-pdf-modal', 'weibull-exponential-combined', language),
    ])


exponential_analysis_page = create_exponential_analysis_page()

def create_rsf_analysis_page(language='es'):
    if language == 'es':
        profile_card_title = 'Simular perfil individual'
        profile_card_note = 'Selecciona un perfil concreto para obtener su curva de supervivencia estimada por RSF.'
        profile_button_label = 'Simular perfil'
        profile_gender_label = 'Género'
        profile_gender_options = [
            {'label': 'Femenino', 'value': 1},
            {'label': 'Masculino', 'value': 0},
        ]
        profile_disability_label = 'Discapacidad'
        profile_disability_options = [
            {'label': 'Con discapacidad', 'value': 1},
            {'label': 'Sin discapacidad', 'value': 0},
        ]
        profile_age_label = 'Edad'
        profile_age_options = [
            {'label': '0-35 años', 'value': 'age_band_0-35'},
            {'label': '35-55 años', 'value': 'age_band_35-55'},
            {'label': '55+ años', 'value': 'age_band_55<='},
        ]
        profile_education_label = 'Nivel educativo'
        profile_education_options = [
            {'label': 'A Level o equivalente', 'value': 'highest_education_A Level or Equivalent'},
            {'label': 'HE Qualification', 'value': 'highest_education_HE Qualification'},
            {'label': 'Inferior a A Level', 'value': 'highest_education_Lower Than A Level'},
            {'label': 'Postgrado', 'value': 'highest_education_Post Graduate Qualification'},
        ]
        profile_credits_label = 'Créditos estudiados'
        profile_credits_options = [
            {'label': 'Pocos créditos (~30)', 'value': 'few'},
            {'label': 'Créditos medios (~60)', 'value': 'medium'},
            {'label': 'Muchos créditos (~120)', 'value': 'many'},
        ]
    else:
        profile_card_title = 'Simulate individual profile'
        profile_card_note = 'Choose a concrete profile to obtain its RSF-estimated survival curve.'
        profile_button_label = 'Simulate profile'
        profile_gender_label = 'Gender'
        profile_gender_options = [
            {'label': 'Female', 'value': 1},
            {'label': 'Male', 'value': 0},
        ]
        profile_disability_label = 'Disability'
        profile_disability_options = [
            {'label': 'With disability', 'value': 1},
            {'label': 'Without disability', 'value': 0},
        ]
        profile_age_label = 'Age band'
        profile_age_options = [
            {'label': '0-35 years', 'value': 'age_band_0-35'},
            {'label': '35-55 years', 'value': 'age_band_35-55'},
            {'label': '55+ years', 'value': 'age_band_55<='},
        ]
        profile_education_label = 'Education level'
        profile_education_options = [
            {'label': 'A Level or Equivalent', 'value': 'highest_education_A Level or Equivalent'},
            {'label': 'HE Qualification', 'value': 'highest_education_HE Qualification'},
            {'label': 'Lower Than A Level', 'value': 'highest_education_Lower Than A Level'},
            {'label': 'Post Graduate Qualification', 'value': 'highest_education_Post Graduate Qualification'},
        ]
        profile_credits_label = 'Studied credits'
        profile_credits_options = [
            {'label': 'Few credits (~30)', 'value': 'few'},
            {'label': 'Medium credits (~60)', 'value': 'medium'},
            {'label': 'Many credits (~120)', 'value': 'many'},
        ]

    return html.Div([
        html.H1(
            get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'random_survival_forest')),
            style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '20px', 'color': '#1a1a1a', 'fontWeight': 'bold'}
        ),

        html.Div([
            html.P(
                get_translation(language, 'rsf_intro'),
                style={'margin': 0, 'fontSize': '1.05em', 'lineHeight': '1.6', 'color': '#34495e'}
            ),
            html.P(
                get_translation(language, 'rsf_note'),
                style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic'}
            ),
        ], style={
            'backgroundColor': '#f8fbff',
            'padding': '22px',
            'borderRadius': '10px',
            'border': '1px solid #dfe9f3',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'margin': '0 20px 25px 20px'
        }),

        html.Div(
            dcc.Loading(
                id='loading-rsf',
                type='circle',
                children=html.Div(id='rsf-analysis-output')
            ),
            style={'margin': '0 20px 30px 20px'}
        ),

        html.Div([
            html.H3(profile_card_title, style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.P(profile_card_note, style={'margin': '0 0 18px 0', 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic', 'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Label(profile_gender_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-gender', options=profile_gender_options, value=1, clearable=False),
                ]),
                html.Div([
                    html.Label(profile_disability_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-disability', options=profile_disability_options, value=1, clearable=False),
                ]),
                html.Div([
                    html.Label(profile_age_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-age-band', options=profile_age_options, value='age_band_0-35', clearable=False),
                ]),
                html.Div([
                    html.Label(profile_education_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-education', options=profile_education_options, value='highest_education_A Level or Equivalent', clearable=False),
                ]),
                html.Div([
                    html.Label(profile_credits_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-credits', options=profile_credits_options, value='few', clearable=False),
                ]),
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))',
                'gap': '14px',
                'marginBottom': '18px'
            }),
            html.Div([
                html.Button(profile_button_label, id='rsf-profile-simulate-btn', style={'padding': '10px 20px', 'backgroundColor': '#8e44ad', 'color': 'white', 'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
            ], style={'textAlign': 'center'}),
            html.Div(id='rsf-profile-output', style={'marginTop': '18px'}),
            html.Div([
                html.Button(get_translation(language, 'explicar_rsf'), id='btn-rsf',
                                                     title=("Primero carga y preprocesa un dataset" if language == 'es' else "Load and preprocess a dataset first"),
                           style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                                 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
                html.Button(f"📄 {'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-rsf-btn',
                                                 title=("Necesitas ejecutar RSF y obtener resultados antes de exportar" if language == 'es' else "Run RSF and get results before exporting"),
                         style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginLeft': '10px'})
            ], style={'textAlign': 'center', 'marginTop': '18px', 'marginBottom': '16px'}),
            html.Div(
                dcc.Textarea(
                    id='openai-answer-rsf',
                    value=get_translation(language, 'respuesta'),
                    style={
                        'width': '90%',
                        'minHeight': '180px',
                        'resize': 'none',
                        'whiteSpace': 'pre-wrap',
                        'margin': '0 auto',
                        'display': 'block',
                        'border': '1px solid #ccc',
                        'borderRadius': '8px',
                        'fontSize': '14px',
                        'padding': '14px',
                        'overflowY': 'auto',
                        'color': '#1a1a1a',
                        'backgroundColor': '#fff'
                    },
                    disabled=True
                ),
                style={'textAlign': 'center'}
            ),
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'margin': '0 20px 30px 20px'
        }),

            create_pdf_export_modal('rsf-pdf-modal', 'rsf', language),
        dcc.Store(id='rsf-analysis-data')
    ])

rsf_analysis_page = create_rsf_analysis_page()

def create_ver_dataset_page(language='es'):
    return html.Div([ 
        html.H1(get_translation(language, 'dataset_limpio'), style={'textAlign': 'center', 'fontSize': '2.5em'}),
        html.Div([ 
            dash_table.DataTable(
                id='clean-dataset-table',
                columns=[{"name": col, "id": col} for col in df.columns],  
                data=df.to_dict('records'),  # convertir el DataFrame en un formato que Dash pueda usar
                style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'},  # Asegurarse de que el texto esté alineado
            ),
        ], style={'textAlign': 'center', 'marginTop': '30px'})
    ])

ver_dataset_page = create_ver_dataset_page()

# Función para crear página de análisis de covariables con soporte multilingüe
# Estilos CSS globales para mejorar interfaz - ahora aplicados inline
# Se pueden usar en los labelStyle de RadioItems

def create_covariate_analysis_page(language='es'):
    return html.Div([
        # Header
        html.Div([
            html.H1(
                '📊 ' + get_translation(language, 'analisis_covariables_title'),
                style={
                    'textAlign': 'center',
                    'fontSize': '2.5em',
                    'color': '#2c3e50',
                    'marginBottom': '10px',
                    'fontWeight': 'bold'
                }
            ),
            html.P(
                get_translation(language, 'analisis_covariables_intro'),
                style={
                    'textAlign': 'center',
                    'fontSize': '1.1em',
                    'color': '#7f8c8d',
                    'marginBottom': '30px'
                }
            )
        ], className='section-header formal-card', style={'backgroundColor': '#f8fbff', 'padding': '30px', 'borderRadius': '10px', 'margin': '20px'}),
        
        # Botones de selección
        html.Div([
            html.Div([
                html.Label(
                    get_translation(language, 'selecciona_el_analisis'),
                    style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': '#2c3e50', 'display': 'block', 'marginBottom': '15px'}
                ),
                dcc.RadioItems(
                    options=[
                        {'label': f"📈 {get_translation(language, 'abandono_total')}", 'value': 'abandono'},
                        {'label': f"👥 {get_translation(language, 'abandono_genero')}", 'value': 'gender'},
                        {'label': f"♿ {get_translation(language, 'abandono_discapacidad')}", 'value': 'disability'},
                        {'label': f"🎂 {get_translation(language, 'abandono_age_band')}", 'value': 'age_band'},
                        {'label': f"🎓 {get_translation(language, 'abandono_highest_education')}", 'value': 'highest_education'},
                        {'label': f"📚 {get_translation(language, 'abandono_studied_credits')}", 'value': 'studied_credits'}
                    ],
                    value='abandono',
                    id='covariables-dropdown',
                    labelStyle={
                        'display': 'inline-block',
                        'padding': '12px 20px',
                        'margin': '8px',
                        'border': '2px solid #ddd',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                        'backgroundColor': '#f8f9fa',
                        'transition': 'all 0.3s ease',
                        'fontWeight': '500',
                        'textAlign': 'center'
                    }
                )
            ], className='formal-card', style={
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'margin': '0 20px'
            })
        ]),
        
        # Contenido principal
        html.Div([
            html.Div([
                # Gráfico
                html.Div([
                    dcc.Loading(
                        id="loading-covariables",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id='covariables-graph',
                                style={'height': '500px'},
                                config={'responsive': True, 'displayModeBar': True}
                            )
                        ]
                    )
                ], className='graph-container', style={
                    'width': 'calc(100% - 340px)',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '20px'
                }),
                
                # Panel de explicación
                html.Div([
                    html.H3(f"💡 {get_translation(language, 'interpretacion')}", style={'color': '#1abc9c', 'marginTop': '0', 'fontWeight': 'bold'}),
                    html.Div(
                        id='graph-explanation',
                        className='explanation-panel',
                        style={'minHeight': '400px', 'overflow': 'auto'}
                    )
                ], className='interpretation-card formal-card', style={
                    'width': '320px',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'backgroundColor': '#f8fbff',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                })
            ], className='covariate-main-layout', style={'padding': '20px', 'display': 'flex', 'gap': '0'})
        ], style={'margin': '20px'})
    ])

# Función para crear página de regresión de Cox con soporte multilingüe
def create_cox_regression_page(language='es'):
    return html.Div([
        html.H1(get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'cox_regression')), style={'textAlign': 'center','fontSize': '35px'}),
    
        html.Div([
            dcc.Dropdown(
                id='covariables-dropdown-cox',
                options=[
                    {'label': get_translation(language, 'genero'), 'value': 'gender_F'},
                    {'label': get_translation(language, 'discapacidad'), 'value': 'disability_N'},
                    {'label': get_translation(language, 'edad_banda'), 'value': 'age_band'},
                    {'label': get_translation(language, 'educacion_mas_alta'), 'value': 'highest_education'},
                    {'label': get_translation(language, 'creditos_estudiados'), 'value': 'studied_credits'}
                ],
                placeholder=get_translation(language, 'elige_covariable_cox'),  
                multi=True,  # Permite seleccionar varias covariables
                style={'width': '60%', 'margin': 'auto'}
            ),
        ], style={'textAlign': 'center', 'marginTop': '30px'}),
        
        html.Div(id='cox-regression-output', style={'textAlign': 'center', 'marginTop': '20px'}),
        
        # Botones de acción
        html.Div([
            html.Button(get_translation(language, 'explicar_cox'), id='btn-cox', 
                     title=("Selecciona covariables y ejecuta Cox antes de explicar" if language == 'es' else "Select covariates and run Cox before generating an explanation"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"📄 {'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-cox-btn', 
                     title=("Requiere resultados Cox vigentes (si cambias dataset, recalcula)" if language == 'es' else "Requires up-to-date Cox results (recalculate after dataset changes)"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
        
        html.Div(
            dcc.Textarea(
                id='openai-answer-cox',
                placeholder=get_translation(language, 'respuesta'),
                style={
                    'width': '60%',  # tamaño
                    'height': '400px',  # altura de la caja de texto
                    'resize': 'none',  
                    'whiteSpace': 'pre-wrap',  # texto se ajuste y no se corte
                    'margin': '20px auto',  # centrado automático
                    'display': 'block',  # en bloque y centrado
                    'border': '1px solid #ccc',  # borde para darle un poco de estilo
                    'borderRadius': '8px',  # bordes redondeados
                    'fontSize': '16px',  # tamaño de la fuente para mayor legibilidad
                    'overflowY': 'auto'
                },
                disabled=True 
            ),
            style={'textAlign': 'center'}  # centrado del contenedor que envuelve el Textarea
        ),
        
        # Modal de exportación a PDF
        create_pdf_export_modal('cox-pdf-modal', 'cox-regression', language),
        
        # Store para mantener datos del análisis actual
        dcc.Store(id='cox-analysis-data')
    ])

cox_regression_page = create_cox_regression_page()

# Función para crear página de Kaplan-Meier con soporte multilingüe
def create_kaplan_meier_page(language='es'):
    return html.Div([
    html.H1(get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'kaplan_meier')), 
                style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '30px', 'color': '#1a1a1a', 'fontWeight': 'bold'}),
        
        # ===== GRÁFICA GLOBAL =====
        html.Div([
            html.H3(f"📊 {'Curva de Kaplan-Meier General' if language == 'es' else 'General Kaplan-Meier curve'}", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold'}),
            html.Div(id='km-global-div', children=plot_kaplan_meier(df), 
                    style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
        ], style={'marginBottom': '40px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'padding': '20px', 'borderRadius': '10px'}),
        
        # ===== BOTONES DE SELECCIÓN =====
        html.Div([
            html.Label(get_translation(language, 'selecciona_covariable_kaplan'), 
                      style={'fontWeight':'bold', 'fontSize': '1.1em', 'marginBottom': '15px', 'display': 'block', 'color': '#1a1a1a'}),
            html.Div([
                html.Button('👥 ' + get_translation(language, 'genero'), id='botonG', 
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button('♿ ' + get_translation(language, 'discapacidad'), id='botonDisc', 
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button('🎂 ' + get_translation(language, 'edad_banda'), id='botonAge', 
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button('🎓 ' + get_translation(language, 'educacion_mas_alta'), id='botonEdu', 
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button('📚 ' + get_translation(language, 'creditos_estudiados'), id='botonCredits', 
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button('❌ ' + get_translation(language, 'ninguna'), id='botonNone', 
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#d4edda', 'border': '2px solid #28a745', 'fontSize': '14px', 'fontWeight': '500', 'color': '#0d3d1a'}),
            ], style={'textAlign': 'center', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        ], style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px', 'padding': '20px', 
                 'backgroundColor': '#f8fbff', 'borderRadius': '10px', 'border': '1px solid #ddd'}),
        
        # ===== GRÁFICA DE COVARIABLE (Solo aparece si se selecciona algo) =====
        html.Div(id='km-cov-div', style={'marginTop': '40px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                                        'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}),
        
        # ===== BOTONES DE ACCIÓN =====
        html.Div([
            html.Button(get_translation(language, 'explicar_kaplan'), id='explicar-btn-kaplan', 
                     title=("Selecciona una covariable para generar una explicación estratificada" if language == 'es' else "Select a covariate to generate a stratified explanation"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"📄 {'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-km-btn', 
                     title=("Para incluir gráfica por grupo, selecciona primero una covariable" if language == 'es' else "To include grouped plot, select a covariate first"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
        
        # ===== TEXTAREA PARA EXPLICACIÓN =====
        html.Div(
            dcc.Textarea(
                id='openai-answer-kaplan',
                placeholder=get_translation(language, 'respuesta'),
                style={
                    'width': '80%',  
                    'height': '400px', 
                    'resize': 'none',  
                    'whiteSpace': 'pre-wrap', 
                    'margin': '20px auto',  
                    'display': 'block',  
                    'border': '1px solid #ccc', 
                    'borderRadius': '8px',  
                    'fontSize': '14px', 
                    'padding': '10px',
                    'overflowY': 'auto',
                    'color': '#1a1a1a'
                },
                disabled=True  
            ),
            style={'textAlign': 'center'}
        ),
        
        # Store para mantener variable actual
        dcc.Store(id='km-current-variable', data=''),
        
        # Modal de exportación a PDF
        create_pdf_export_modal('km-pdf-modal', 'kaplan-meier', language)
    ])

kaplan_meier_page = create_kaplan_meier_page()

#Función para crear página de análisis de log-rank con soporte multilingüe
def create_log_rank_page(language='es'):
    return html.Div([
        html.H1(get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'log_rank_test')), style={'textAlign': 'center', 'fontSize': '35px'}),
        
        html.Div([
            dcc.Dropdown(
                id='covariables-dropdown-logrank',
                options=[
                    {'label': get_translation(language, 'genero'), 'value': 'gender_F'},
                    {'label': get_translation(language, 'discapacidad'), 'value': 'disability_N'},
                    {'label': get_translation(language, 'edad_banda'), 'value': 'age_band'},
                    {'label': get_translation(language, 'educacion_mas_alta'), 'value': 'highest_education'},
                    {'label': get_translation(language, 'creditos_estudiados'), 'value': 'studied_credits'}
                ],
                placeholder=get_translation(language, 'selecciona_covariable_logrank'),  
                value=[], 
                multi=True,
                style={'width': '60%', 'margin': 'auto'}
            ),
        ], style={'textAlign': 'center', 'marginTop': '30px'}),

        # Este div se actualizará con el resultado del Log-Rank Test
        html.Div(id='logrank-test-output', style={'textAlign': 'center', 'marginTop': '20px'}),
        
        # Botones de acción
        html.Div([
            html.Button(get_translation(language, 'explicar_logrank'), id='explicar-btn-logrank', 
                     title=("Selecciona covariables y ejecuta Log-Rank antes de explicar" if language == 'es' else "Select covariates and run Log-Rank before generating an explanation"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"📄 {'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-logrank-btn', 
                     title=("Requiere resultados Log-Rank vigentes (si cambias dataset, recalcula)" if language == 'es' else "Requires up-to-date Log-Rank results (recalculate after dataset changes)"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

        html.Div(
            dcc.Textarea(
                id='openai-answer-logrank',
                placeholder=get_translation(language, 'respuesta'),
                style={
                    'width': '60%',  
                    'height': '400px',  
                    'resize': 'none',  
                    'whiteSpace': 'pre-wrap',  
                    'margin': '20px auto', 
                    'display': 'block',  
                    'border': '1px solid #ccc',
                    'borderRadius': '8px',  
                    'fontSize': '16px', 
                    'overflowY': 'auto'
                },
                disabled=True  
            ),
            style={'textAlign': 'center'} 
        ),
        
        # Modal de exportación a PDF
        create_pdf_export_modal('logrank-pdf-modal', 'log-rank', language),
        
        # Store para mantener datos del análisis actual
        dcc.Store(id='logrank-analysis-data')
    ])

log_rank_page = create_log_rank_page()

def display_logrank_summary_table(result):
    df_show = result[['Covariable','Grupo A','Grupo B',
                      'n A','n B','test_statistic','p_value','-log2(p)',
                      'Decisión','Conclusión']].copy()

    return dash_table.DataTable(
        id='logrank-summary-table',
        columns=[{"name": col, "id": col} for col in df_show.columns],
        data=df_show.to_dict('records'),
        style_table={'height': '150px', 'overflowY': 'auto', 'width': '58%', 'margin': 'auto'},
        style_cell={'textAlign': 'center', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_data_conditional=[
            {'if': {'filter_query': '{decision} = "Rechazar H0"'}, 'backgroundColor': '#ffecec'},
            {'if': {'filter_query': '{decision} = "No rechazar H0"'}, 'backgroundColor': '#ecffec'},
        ]
    )

