import os
import platform as _platform
from collections import namedtuple

# Workaround para Python 3.13 en algunos Windows: platform.system() puede
# bloquearse por una consulta WMI al importar Werkzeug/Dash.
if os.name == 'nt':
    _Uname = namedtuple('Uname', ['system', 'node', 'release', 'version', 'machine', 'processor'])
    _fake_uname = _Uname('Windows', 'localhost', '10', '10.0', 'AMD64', 'AMD64')
    _platform.uname = lambda: _fake_uname
    _platform.system = lambda: _fake_uname.system
    _platform.machine = lambda: _fake_uname.machine
    _platform.processor = lambda: _fake_uname.processor

import dash
from dash import Dash, dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output, State
from layout import (
    create_survival_analysis_page, create_covariate_analysis_page, 
    create_kaplan_meier_page, create_cox_regression_page, 
    create_log_rank_page, create_ver_dataset_page, 
    create_weibull_analysis_page, create_exponential_analysis_page,
    create_rsf_analysis_page,
    create_techniques_comparison_page
)
from dash import callback_context
from translations import get_translation
from pdf_callbacks import register_pdf_export_callbacks  # PDF export HABILITADO
from analysis_callbacks import create_home_page, register_analysis_callbacks

# Inicializar la aplicación Dash
app = Dash(__name__, suppress_callback_exceptions=True)
app.config['suppress_callback_exceptions'] = True


# Barra de navegación fija en la parte superior
navbar = html.Div([
    html.Div([ 
        html.Div([
            html.Button(get_translation('es', 'navbar_home'), id='inicio-btn', n_clicks=0, className='navbar-link', style={'border': 'none', 'background': 'none', 'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '14px'}),
            dcc.Link(get_translation('es', 'navbar_view_dataset'), href='/ver-dataset', className='navbar-link', id='navbar-view-dataset'), 
            dcc.Link(get_translation('es', 'navbar_covariate_analysis'), href='/covariate-analysis', className='navbar-link', id='navbar-covariate-analysis'),
            dcc.Link(get_translation('es', 'navbar_survival_analysis'), href='/survival-analysis', className='navbar-link', id='navbar-survival-analysis'),
            dcc.Link(
                get_translation('es', 'navbar_techniques_comparison'),
                href='/survival-analysis/comparacion-tecnicas',
                className='navbar-link',
                id='navbar-techniques-comparison',
                style={'display': 'none'}
            ),
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
     Output('navbar-survival-analysis', 'children'),
     Output('navbar-techniques-comparison', 'children')],
    Input('language-store', 'data')
)
def update_navbar_labels(language):
    return (
        get_translation(language, 'navbar_home'),
        get_translation(language, 'navbar_view_dataset'),
        get_translation(language, 'navbar_covariate_analysis'),
        get_translation(language, 'navbar_survival_analysis'),
        get_translation(language, 'navbar_techniques_comparison'),
    )


@app.callback(
    Output('navbar-techniques-comparison', 'style'),
    [Input('url', 'pathname')]
)
def toggle_techniques_comparison_nav(pathname):
    if pathname and pathname.startswith('/survival-analysis'):
        return {'display': 'inline-block'}
    return {'display': 'none'}

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

    if pathname in ['/ver-dataset', '/covariate-analysis', '/survival-analysis', '/survival-analysis/kaplan-meier', '/survival-analysis/cox-regression', '/survival-analysis/log-rank', '/survival-analysis/weibull', '/survival-analysis/exponential', '/survival-analysis/rsf', '/survival-analysis/comparacion-tecnicas'] and not dataset_loaded:
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
    elif pathname == '/survival-analysis/comparacion-tecnicas':
        return create_techniques_comparison_page(language)
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


if __name__ == '__main__':
    print('[BOOT] Registrando callbacks principales...', flush=True)
    register_analysis_callbacks(app)

    print('[BOOT] Registrando callbacks de exportación PDF...', flush=True)
    register_pdf_export_callbacks(app)  # PDF export HABILITADO

    host = '127.0.0.1'
    port = 8050
    debug_mode = os.environ.get('DASH_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')

    print(f'[BOOT] Iniciando Dash en http://{host}:{port} (debug={debug_mode}, reloader=False)', flush=True)
    app.run_server(host=host, port=port, debug=debug_mode, use_reloader=False)
