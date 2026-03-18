import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output  
import plotly.express as px
import base64
import io
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc 
import numpy as np
from translations import get_translation

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


# Funciones para crear páginas dinámicamente con soporte multilingüe
def create_survival_analysis_page(language='es'):
    return html.Div([  
        html.H1(get_translation(language, 'survival_analysis'), style={'textAlign': 'center'}),
        # Barra de navegación interna
        html.Div([
            html.Div([
                html.Img(
                    src='/assets/kaplan.png', 
                    style={'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px','marginBottom': '15px'}
                ),
                dcc.Link(get_translation(language, 'kaplan_meier'), href='/survival-analysis/kaplan-meier', className='home-link'),

            ], style={'textAlign': 'center',  'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%'}),

            html.Div([
                html.Img(
                    src='/assets/cox.png', 
                    style={'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px','marginBottom': '30px'}  
                ),
                dcc.Link(get_translation(language, 'cox_regression'), href='/survival-analysis/cox-regression', className='home-link'),
            ], style={'textAlign': 'center',  'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%'}),  

            html.Div([
                html.Img(
                    src='/assets/logrank.png', 
                    style={'width': '50%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px','marginBottom': '30px'} 
                ),
                dcc.Link(get_translation(language, 'log_rank_test'), href='/survival-analysis/log-rank', className='home-link'),
            ], style={'textAlign': 'center',  'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%'}),  
        ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'gap': '30px', 'marginTop': '20px'}),  

        # Contenedor para mostrar el gráfico de Kaplan-Meier
        html.Div(id='survival_analysis_page', children=[]),
    ])

# Páginas predefinidas (se actualizarán dinámicamente con callbacks)
survival_analysis_page = create_survival_analysis_page()

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
def create_covariate_analysis_page(language='es'):
    return html.Div([
        html.Div(className='row', children=[
            html.H1(get_translation(language, 'analisis_covariables_title'), style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'gap': '30px', 'marginTop': '20px'}),
        ]),
        html.Div(className='row', children=[
            dcc.RadioItems(
                options=[
                    {'label': get_translation(language, 'abandono_total'), 'value': 'abandono'},
                    {'label': get_translation(language, 'abandono_genero'), 'value': 'gender'},
                    {'label': get_translation(language, 'abandono_discapacidad'), 'value': 'disability'},
                    {'label': get_translation(language, 'abandono_age_band'), 'value': 'age_band'},
                    {'label': get_translation(language, 'abandono_highest_education'), 'value': 'highest_education'},
                    {'label': get_translation(language, 'abandono_studied_credits'), 'value': 'studied_credits'}
                ],
            value='abandono',
                id='covariables-dropdown',
                style={'width': '80%', 'padding': '10px', 'margin': 'auto', 'display': 'block'}
            )
        ], style={'textAlign': 'center', 'marginTop': '30px'}),
        
        html.Div(className='row', children=[
            html.Div(className='six columns', children=[
                dcc.Graph(id='covariables-graph')  
            ], style={'width': '70%', 'padding': '10px', 'display': 'inline-block'}),  

            html.Div(className='six columns', children=[  
                html.Div(id='graph-explanation', style={'padding': '10px', 'display': 'flex', 'alignItems': 'center'})
            ],style={
                'position': 'fixed',        #  barra en el lado derecho
                'top': '7%',               
                'right': '0',               # Pegado al margen derecho
                'z-index': '1000',          # se superponga sobre otros elementos
                'backgroundColor': '#f4f7f6',  # fondo diferenciado de la página
                'width': '300px',           # ancho de la barra lateral
                'height': '100vh',  # la barra ocupe todo el alto de la pantalla
                'borderLeft': '2px solid #ccc',  # borde para diferenciarla
                'padding': '10px',
                'display': 'flex', 
                'color': '#6A6A6A',
                'flexDirection': 'column',  # contenido dentro centrado
                'justifyContent': 'center'
            })
        ], style={'display': 'flex', 'alignItems': 'center'})
    ])

covariate_analysis_page = create_covariate_analysis_page()

# Función para crear página de regresión de Cox con soporte multilingüe
def create_cox_regression_page(language='es'):
    return html.Div([
        html.H1(f"Survival Analysis: {get_translation(language, 'cox_regression')}", style={'textAlign': 'center','fontSize': '35px'}),
    
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
        html.Button(get_translation(language, 'explicar_cox'), id='btn-cox', style={'margin-top': '20px', 'display': 'block', 'margin': 'auto'}),
    html.Div(
            dcc.Textarea(
                id='openai-answer-cox',
                placeholder=get_translation(language, 'respuesta'),
                style={
                    'width': '60%',  #  tamaño
                    'height': '400px',  # altura de la caja de texto
                    'resize': 'none',  
                    'whiteSpace': 'pre-wrap',  # texto se ajuste y no se corte
                    'margin': '20px auto',  # centrado automático
                    'display': 'block',  # en bloque y centrado
                    'border': '1px solid #ccc',  # borde para darle un poco de estilo
                    'borderRadius': '8px',  # bordes redondeados
                    'fontSize': '16px',  #  tamaño de la fuente para mayor legibilidad
                    'overflowY': 'auto'
                },
                disabled=True 
            ),
            style={'textAlign': 'center'}  # centrado del contenedor que envuelve el Textarea
        ) 
    ])

cox_regression_page = create_cox_regression_page()

# Función para crear página de Kaplan-Meier con soporte multilingüe
def create_kaplan_meier_page(language='es'):
    return html.Div([
        html.H1(f"Survival Analysis: {get_translation(language, 'kaplan_meier')}", style={'textAlign': 'center', 'fontSize': '35px'}),
    
    # 1) Gráfica global
    html.Div(id='km-global-div', children=plot_kaplan_meier(df)),
    
    html.Div([
        html.Label(get_translation(language, 'selecciona_covariable_kaplan'), style={'fontWeight':'bold'}),
        html.Div([
            html.Button(get_translation(language, 'genero'), id='botonG', style={'border-radius': '50%', 'padding': '10px 20px', 'margin': '10px'}),
            html.Button(get_translation(language, 'discapacidad'), id='botonDisc', style={'border-radius': '50%', 'padding': '10px 20px', 'margin': '10px'}),
            html.Button(get_translation(language, 'ninguna'), id='botonNone', style={'border-radius': '50%', 'padding': '10px 20px', 'margin': '10px', 'background-color': '#accfc3'}),

        ], style={'textAlign': 'center'}),
    ], style={'textAlign': 'center','marginTop':'30px'}),

    # 3) Div donde irá la gráfica de la covariable
    html.Div(id='km-cov-div', style={'marginTop':'20px'}),
    html.Button(get_translation(language, 'explicar_kaplan'), id='explicar-btn-kaplan', style={'margin-top': '20px', 'display': 'block', 'margin': 'auto'}),

    html.Div(
        dcc.Textarea(
            id='openai-answer-kaplan',
            placeholder='La respuesta es...',
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
    )
    ])

kaplan_meier_page = create_kaplan_meier_page()

#Función para crear página de análisis de log-rank con soporte multilingüe
def create_log_rank_page(language='es'):
    return html.Div([
    html.H1(f"Survival Analysis: {get_translation(language, 'log_rank_test')}", style={'textAlign': 'center', 'fontSize': '35px'}),
    
    html.Div([
        #html.Label("Selecciona 1 o más covariables para el Test de Log-Rank:", style={'fontWeight':'bold'}),
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
            multi= True,
            style={'width': '60%', 'margin': 'auto'}
        ),
    ], style={'textAlign': 'center', 'marginTop': '30px'}),

    # Este div se actualizará con el resultado del Log-Rank Test
    html.Div(id='logrank-test-output', style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Button(get_translation(language, 'explicar_logrank'), id='explicar-btn-logrank', style={'margin-top': '20px', 'display': 'block', 'margin': 'auto'}),

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
    )
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

