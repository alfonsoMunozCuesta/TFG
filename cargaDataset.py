from dash import Dash,dash, dcc, html, dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import base64
import plotly.express as px
import io
from layout import (
    create_survival_analysis_page, create_covariate_analysis_page, 
    create_kaplan_meier_page, create_cox_regression_page, 
    create_log_rank_page, create_ver_dataset_page, 
    display_logrank_summary_table
)
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc
from cox_regression import run_cox_regression
from log_rank_test import perform_log_rank_test
import matplotlib.pyplot as plt
import requests
from ollama_AI import generate_explanation 
import plotly.graph_objs as go
from dash import callback_context
from translations import get_translation

# Inicializar la aplicación Dash
app = Dash(__name__, suppress_callback_exceptions=True)

# Configuración de llama-server (llama.cpp) con Qwen2.5-1.5B-Instruct
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

df = pd.read_csv(r"C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis\data\temp_data.csv", sep=';')
df_limpio = pd.read_csv(r'C:\Users\LENOVO\Desktop\CODE_LUCI\Survival-Analysis\dataset_limpio.csv', sep=';')


# Barra de navegación fija en la parte superior
navbar = html.Div([
    html.Div([ 
        html.Div([
            html.Button('INICIO', id='inicio-btn', n_clicks=0, className='navbar-link', style={'border': 'none', 'background': 'none', 'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '14px'}),
            dcc.Link('VER DATASET', href='/ver-dataset', className='navbar-link'), 
            dcc.Link('ANÁLISIS DE COVARIABLES', href='/covariate-analysis', className='navbar-link'),
            dcc.Link('ANÁLISIS DE SUPERVIVENCIA', href='/survival-analysis', className='navbar-link'),
        ], className='navbar-links', style={'flex': '1'}),
        # Botón de idioma en la esquina derecha
        html.Div([
            dcc.RadioItems(
                id='language-selector',
                options=[
                    {'label': 'ES', 'value': 'es'},
                    {'label': 'EN', 'value': 'en'},
                ],
                value='es',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                style={
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'padding': '5px 15px',
                    'borderRadius': '5px',
                    'backgroundColor': 'rgba(26, 188, 156, 0.05)',
                }
            )
        ], style={'marginRight': '30px', 'display': 'flex', 'alignItems': 'center'})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'width': '100%'})
], id='navbar', style={'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'})

app.layout = html.Div([ 
    navbar, 
    dcc.Location(id='url', refresh=False),  
    dcc.Store(id='language-store', data='es'),  # Store para almacenar el idioma actual
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
    Input('language-selector', 'value')
)
def update_language(selected_language):
    return selected_language

@app.callback(
    Output('confirm-dialog', 'displayed'),
    Input('inicio-btn', 'n_clicks'),
    prevent_initial_call=True  # Evita que se active al cargar la página
)
def mostrar_confirmacion(n_clicks):
    if n_clicks:
        return True 
    return False

@app.callback(
    Output('confirm-dialog', 'message'),
    Input('language-store', 'data')
)
def update_confirm_message(language):
    return get_translation(language, 'confirmar_inicio')

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
    #Aceptar
    if 'submit_n_clicks' in trigger_id:
        print("Aceptar clickeado")
        return '/'  # Redirige a la página de inicio
    
    # Cancelar"
    if 'cancel_n_clicks' in trigger_id:
        print("Cancelar clickeado")
        return dash.no_update  # No hacer nada
    
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
            html.Img(src='/assets/datos.png', style={'height': '150px', 'marginTop': '10px', 'marginLeft': '20px', 'display': 'none'}), 
            html.H1(get_translation(language, 'dashboard_title'), style={'textAlign': 'center', 'fontSize': '2.5em', 'display': 'inline-block'})
        ], style={'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '0px'}), 
        
        dcc.Loading(
            id="loading-spinner",
            type="circle",  
            children=html.Div([  
                html.H3(get_translation(language, 'cargar_dataset'), id='upload-text'), 
                dcc.Upload(
                    id='upload-data',
                    children=html.Button(get_translation(language, 'sube_csv')),
                    multiple=False
                ),
                html.Div(id='output-data-upload') 
            ], style={'textAlign': 'center', 'marginTop': '30px'}),
        ),
        
        # Botón para limpiar y cargar el dataset limpio
        html.Div([ 
            html.Button(get_translation(language, 'preprocesa_csv'), id='load-clean', n_clicks=0),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
    ])

# Página inicial
home_page = create_home_page('es')
#ocultar frase incial: "Cargar Dataset..."
@app.callback(
    Output('upload-text', 'style'),  # Cambiar el estilo del texto
    [Input('upload-data', 'contents')]
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
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=";")
    return df.head(10000)  # Limitamos a las primeras 10000 filas por eficiencia

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
     Output('output-data-upload', 'children')], 
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'), 
     Input('load-clean', 'n_clicks')],
    [State('language-store', 'data')]
)

def update_output(contents, filename, n_clicks, language):
    if contents is None:
        return {'display': 'block'},{'display': 'none'}, html.Div([get_translation(language, 'no_archivo_cargado')], style={'marginTop': '20px', 'marginBottom': '0px'})
    
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
        )
    # Cargar el archivo CSV
    df = parse_contents(contents)
    
    if n_clicks > 0:
        return {'display': 'none'}, {'display': 'none'}, display_data(df_limpio, get_translation(language, 'archivo_preprocesado'))
    
    # Si no se ha presionado el botón de limpiar, mostrar el archivo bruto
    return {'display': 'none'}, {'display': 'inline-block'}, display_data(df, get_translation(language, 'archivo_bruto'))


#maneja que no aparezca la barra de navegacion hasta que se limpie el dataset
@app.callback(
    Output('navbar', 'style'),
    [Input('load-clean', 'n_clicks')]
)
def toggle_navbar(n_clicks):
    if n_clicks > 0:
        return {'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'}
    return {'display': 'none'}  # Si no se ha presionado el botón, la barra de navegación permanece oculta

# Callbacks para manejar la navegación entre páginas
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('language-store', 'data')]
)
def display_page(pathname, language):
    if pathname == '/':
        return create_home_page(language)
    elif pathname == '/covariate-analysis':
        return create_covariate_analysis_page(language)
    elif pathname == '/survival-analysis':
        return create_survival_analysis_page(language)
    elif pathname == '/survival-analysis/kaplan-meier':
        return create_kaplan_meier_page(language)
    elif pathname == '/survival-analysis/cox-regression':
        return create_cox_regression_page(language)
    elif pathname == '/survival-analysis/log-rank':
        return create_log_rank_page(language)
    elif pathname == '/ver-dataset':  
        return create_ver_dataset_page(language)
    else:
        return create_home_page(language)

#maneja navegacion de kaplan
@app.callback(
    Output('km-cov-div', 'children'),
    [Input('botonG', 'n_clicks'), Input('botonDisc', 'n_clicks'), Input('botonNone', 'n_clicks')]
)
def update_km_cov(gender_clicks, disability_clicks, none_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return None  
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'botonG':
        return plot_km_G(df_limpio)  
    elif button_id == 'botonDisc':
        return plot_km_disc(df_limpio, group_by='disability_N') 
    elif button_id == 'botonNone':
        return None 

df_limpio['gender'] = df_limpio['gender_F'].map({1: 'Femenino', 0: 'Masculino'})
df_limpio['disability'] = df_limpio['disability_N'].map({1: 'Con Discapacidad', 0: 'Sin Discapacidad'})
df_limpio['abandono'] = df_limpio['final_result'].map({1: 'Abandono', 0: 'No abandono'})

@app.callback(
    Output('openai-answer-kaplan', 'value'),
    [Input('explicar-btn-kaplan', 'n_clicks')],
    [State('km-global-div', 'children'),
     State('language-store', 'data')] 
)
def explicar_kaplan(n_clicks, kaplan_img, language):

    if n_clicks is not None and n_clicks > 0:
        # Verificar que la imagen de Kaplan-Meier ha sido generada antes de continuar
        if kaplan_img:
            prompt_template = "Give me a conclusion in {lang} of the Kaplan-Meier graphs obtained: {kaplan_img}." if language == 'en' else "Dame una conclusión en español de las gráficas obtenidas de Kaplan-Meier: {kaplan_img}."
            prompt = prompt_template.format(lang='English', kaplan_img=kaplan_img)
            # Llamar a la IA para obtener la explicación
            respuesta = responder_pregunta_con_llama3(prompt)
            return respuesta
    return ""  # Si no se hace clic o no hay gráfica, retornar vacío

# Callback para actualizar el gráfico según la selección del Dropdown
@app.callback(
    [Output('covariables-graph', 'figure'),
    Output('graph-explanation', 'children')],
    [Input('covariables-dropdown', 'value'),
     Input('language-store', 'data')]
)
def update_graph(col_chosen, language):
    if col_chosen == 'abandono':
        conteo_abandono = df_limpio['final_result'].value_counts().sort_index()

        # Crear el gráfico con barras
        fig = go.Figure()

        # Contar las cantidades para cada categoría (No Abandono y Abandono)
        count_no_abandono = conteo_abandono[0] if 0 in conteo_abandono else 0
        count_abandono = conteo_abandono[1] if 1 in conteo_abandono else 0

        # Etiquetas en el idioma seleccionado
        label_no_abandono = get_translation(language, 'no_abandono')
        label_abandono = get_translation(language, 'abandono')

        # Añadir las barras para No Abandono y Abandono
        fig.add_trace(go.Bar(
            x=[label_no_abandono],  
            y=[count_no_abandono],  
            name='',
            marker_color='#1abc9c',
            hovertemplate=f'{label_no_abandono}: %{{y}}'
        ))

        fig.add_trace(go.Bar(
            x=[label_abandono],  
            y=[count_abandono],  
            name='',
            marker_color='#006400',
            hovertemplate=f'{label_abandono}: %{{y}}'
        ))

        # Personalizar la apariencia del gráfico
        fig.update_layout(
            title=get_translation(language, 'abandono_vs_no_abandono'),
            xaxis_title=get_translation(language, 'resultado_final'),
            yaxis_title=get_translation(language, 'num_estudiantes'),
            barmode='group', 
            xaxis=dict(
                tickmode='array', 
                tickvals=[label_no_abandono, label_abandono],  
                ticktext=[f'{label_no_abandono}: {count_no_abandono}', f'{label_abandono}: {count_abandono}'], 
            ),
            legend_title=get_translation(language, 'evento'),
        )
    
        explicacion = get_translation(language, 'exp_abandono')
           
        return fig, explicacion
    
    elif col_chosen == 'gender':
        conteo_genero = df_limpio['gender'].value_counts().sort_index()
        
        # Etiquetas en el idioma seleccionado
        label_masculino = get_translation(language, 'masculino')
        label_femenino = get_translation(language, 'femenino')
        
        # Contamos cuántos estudiantes hay según su género y si han abandonado o no
        fig = px.histogram(df_limpio, x='gender', color='final_result', barmode='group', 
                           title=get_translation(language, 'abandono_genero_title'),
                           color_discrete_map={0: '#1abc9c', 1: '#006400'})

        ticktext = [f'{label_masculino}: {conteo_genero[0]}', f'{label_femenino}: {conteo_genero[1]}']

        fig.update_layout(
            xaxis_title=get_translation(language, 'genero'), 
            yaxis_title=get_translation(language, 'num_estudiantes'), 
            legend_title=get_translation(language, 'abandono'),
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=ticktext)  
        )
        
        explicacion = get_translation(language, 'exp_genero')
        return fig, explicacion
    
    elif col_chosen == 'disability':
        conteo_discapacidad = df_limpio['disability'].value_counts().sort_index()

        # Etiquetas en el idioma seleccionado
        label_sin_discapacidad = get_translation(language, 'sin_discapacidad')
        label_con_discapacidad = get_translation(language, 'con_discapacidad')
        
        # Contamos cuántos estudiantes hay según si tienen discapacidad y si han abandonado o no
        fig = px.histogram(df_limpio, x='disability', color='final_result', barmode='group', 
                           title=get_translation(language, 'abandono_discapacidad_title'),
                           color_discrete_map={0: '#1abc9c', 1: '#006400'})

        ticktext = [f'{label_sin_discapacidad}: {conteo_discapacidad[0]}', f'{label_con_discapacidad}: {conteo_discapacidad[1]}']

        fig.update_layout(
            xaxis_title=get_translation(language, 'discapacidad'), 
            yaxis_title=get_translation(language, 'num_estudiantes'), 
            legend_title=get_translation(language, 'abandono'),
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=ticktext) 
        )

        explicacion = get_translation(language, 'exp_discapacidad')
        return fig, explicacion

@app.callback(
    Output('cox-regression-output', 'children'),
    [Input('covariables-dropdown-cox', 'value'),
     Input('language-store', 'data')]
)
def update_cox_model(covariables, language):
    if covariables is None or len(covariables) == 0:
        return html.Div([get_translation(language, 'selecciona_covariable_minimo')])

    # Asegurarnos de que covariables sea una lista
    if isinstance(covariables, str):  
        covariables = [covariables]
        
    # Llamamos a la función de regresión de Cox con las covariables seleccionadas
    summary, cox_table_html = run_cox_regression(df_limpio, covariables)
    
    return cox_table_html

@app.callback(
    Output('openai-answer-cox', 'value'),
    [Input('btn-cox', 'n_clicks')],
    [State('cox-regression-output', 'children'),
     State('language-store', 'data')] 
)
def explicar_cox(n_clicks, cox_content, language):
    if n_clicks is not None and n_clicks > 0:
        # Verificar que la tabla de Cox ha sido generada antes de continuar
        if cox_content:
            if language == 'en':
                prompt = (
                    f"Explain the results of the Cox Regression generated with the following results:\n"
                    f"Results: {cox_content}\n"
                    f"How do the covariates affect the probability of dropout?"
                )
            else:
                prompt = (
                    f"Explica los resultados dela Regresion de Cox generados con los siguientes resultados:\n"
                    f"Resultados: {cox_content}\n"
                    f"¿Cómo afectan las covariables a la probabilidad de abandono?"
                )
            # Llamar a la IA para obtener la explicación
            respuesta = responder_pregunta_con_llama3(prompt)
            return respuesta
    return "" 

#callback para el control del analisis de log rank
@app.callback(
    Output('logrank-test-output', 'children'),
    [Input('covariables-dropdown-logrank', 'value'),
     Input('language-store', 'data')]
)

def update_logrank_test(covariables, language):
    # Verificar que al menos se haya seleccionado una covariable
    if not covariables:
        return html.Div([get_translation(language, 'selecciona_covariable_comparar')])

    panels = []

    # Ejecutar el Test de Log-Rank para cada covariable seleccionada
    for covariable in covariables:
        res_df = perform_log_rank_test(df_limpio, covariable)
        table = display_logrank_summary_table(res_df)
        panels.append(html.Div([html.H3(f"Resultado del Test de Log-Rank para {covariable}",
                                        style={'textAlign': 'center'}), table]))
        
    # Mostrar todos los resultados para las covariables seleccionadas
    return html.Div(panels)

@app.callback(
    Output('openai-answer-logrank', 'value'),
    [Input('explicar-btn-logrank', 'n_clicks')],
    [State('logrank-test-output', 'children'),
     State('language-store', 'data')]  
)
def explicar_logrank(n_clicks, logrank_content, language):
    if n_clicks is not None and n_clicks > 0:
        # Verificar que la tabla del Test de Log-Rank ha sido generada antes de continuar
        if logrank_content:
            if language == 'en':
                prompt = (
                    f"Can you give me a brief conclusion based on the results obtained from Log Rank:\n"
                    f"Results: {logrank_content}\n"
                )
            else:
                prompt = (
                    #RESPUESTA LARGA
                    #f"Explica los resultados del Test de Log-Rank generados con los siguientes resultados:\n"
                    #RESPUESTA CORTA
                    f"Me puedes dar una conclusión breve según los resultados obtenidos de Log Rank:\n"
                    f"Resultados: {logrank_content}\n"
                    #f"¿Qué significa la diferencia entre los dos grupos y cómo deben interpretarse los valores p y el estadístico de prueba?"
                )
            # Llamar a la IA para obtener la explicación
            respuesta = responder_pregunta_con_llama3(prompt)
            if len(respuesta) > 3000:  # Si la respuesta es muy larga, la cortamos en partes
                chunks = [respuesta[i:i + max_length] for i in range(0, len(respuesta), max_length)]
                return '\n\n'.join(chunks)  # Devolvemos las partes concatenadas
            return respuesta
    return ""


# Callback para las consultas al modelo Qwen2.5 vía llama-server
def responder_pregunta_con_llama3(pregunta: str) -> str:
    """
    Envía la pregunta a Qwen2.5-1.5B-Instruct mediante llama.cpp (llama-server).
    Usa endpoint HTTP local OpenAI-compatible sin dependencias externas.
    Tiempo ilimitado para que el modelo piense.
    """
    import time
    
    try:
        # Payload compatible con API OpenAI de llama-server
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": pregunta}
            ],
            "temperature": 0.2,
            "max_tokens": 3000,
            "stream": False
        }
        
        # Registro de inicio
        inicio = time.time()
        print(f"\n⏳ Enviando solicitud al LLM...")
        print(f"🔄 Esperando respuesta (sin límite de tiempo)...")
        
        # Realiza la solicitud HTTP al servidor llama.cpp (timeout sin límite: 10 minutos)
        response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=600)
        response.raise_for_status()
        
        # Calcular tiempo de respuesta
        tiempo_respuesta = time.time() - inicio
        
        # Extrae la respuesta generada
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        if not content:
            raise ValueError("No se recibió una respuesta válida del modelo.")
        
        # Mostrar tiempo de procesamiento
        minutos = int(tiempo_respuesta // 60)
        segundos = int(tiempo_respuesta % 60)
        tiempo_str = f"{minutos}m {segundos}s" if minutos > 0 else f"{segundos}s"
        print(f"\n✅ Respuesta generada en: {tiempo_str}")
        print(f"📝 Tokens recibidos: {result.get('usage', {}).get('completion_tokens', 'desconocido')}\n")
        
        max_length = 3000
        if len(content) > max_length:
            content_parts = [content[i:i+1000] for i in range(0, len(content), 1000)]
            return '\n\n'.join(content_parts)
        
        return content
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: No se pudo conectar a llama-server. Asegúrate de que está ejecutándose en http://127.0.0.1:8000"
    except requests.exceptions.Timeout:
        return "❌ Error: Timeout (el modelo tardó demasiado en responder, >10 minutos). Intenta con prompts más cortos."
    except Exception as e:
        return f"❌ Error consultando Qwen2.5: {str(e)}"
    

# Correr la app
if __name__ == '__main__':
    app.run_server(debug=True)
