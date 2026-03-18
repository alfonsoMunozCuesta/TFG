from dash import dcc
import plotly.graph_objs as go
from lifelines import KaplanMeierFitter
from config import df_limpio

def plot_kaplan_meier(df):
    kmf = KaplanMeierFitter()

    kmf.fit(df['date'], event_observed=df['final_result'])

    # Crear la figura con Plotly
    fig = go.Figure()

    # curva de Kaplan-Meier
    fig.add_trace(go.Scatter(
        x=kmf.timeline.tolist(),
        y=kmf.survival_function_['KM_estimate'].tolist(),
        mode='lines',
        name='',
        line=dict(color='blue', width=2),
        hovertemplate="(%{x:.0f}, %{y:.3f})" 

    ))

    fig.update_layout(
        title="Curva de Supervivencia Kaplan-Meier",
        xaxis_title="Tiempo",
        yaxis_title="Probabilidad de Supervivencia",
        yaxis=dict(range=[0, 1]),

    )

    return dcc.Graph(figure=fig)


def plot_km_G(df, group_by='gender_F'):
    kmf = KaplanMeierFitter()

    # filtrar datos por género 
    df_gender = df[df[group_by].notnull()]
    fig = go.Figure()

    # Mapeo para convertir 1 y 0 a "Female" y "Male"
    gender_map = {1: "Femenino", 0: "Masculino"}

    # Graficar para cada grupo de género (0 y 1)
    for group in df_gender[group_by].unique():
        # Obtener la etiqueta correspondiente (Female o Male)
        gender_label = gender_map.get(group, "Unknown")
        
        # Ajustar el modelo Kaplan-Meier para cada grupo
        kmf.fit(df_gender[df_gender[group_by] == group]['date'],
                event_observed=df_gender[df_gender[group_by] == group]['final_result'],
                label=f"{gender_label}")
        
        # Obtener el nombre de la columna de probabilidades de supervivencia
        survival_col = kmf.survival_function_.columns[0]

        # Añadir la curva de Kaplan-Meier con bandas de confianza al gráfico
        fig.add_trace(go.Scatter(
            x=kmf.timeline.tolist(),
            y=kmf.survival_function_[survival_col].tolist(),
            mode='lines',
            name=f"{gender_label}",
            line=dict(width=2),
            fill='tonexty',  # Rellenar el área hacia abajo de la curva
            fillcolor='rgba(0, 123, 255, 0.2)' if gender_label == 'Femenino' else 'rgba(255, 165, 0, 0.2)',  # Colores con transparencia para el relleno
            hovertemplate="(%{x:.0f}, %{y:.3f})"
        ))

    # Configurar la gráfica
    fig.update_layout(
        title=f'Curva de Kaplan-Meier para la covariable {group_by}',
        xaxis_title='Tiempo',
        yaxis_title='Probabilidad de Supervivencia',
        yaxis=dict(range=[0, 1]),
        legend_title=group_by,
        template='plotly_white', 
    )

    return dcc.Graph(figure=fig)

def plot_km_disc(df, group_by='disability_N'):
    kmf = KaplanMeierFitter()

    # Filtrar datos por discapacidad y ajustar el modelo de Kaplan-Meier
    df_disability = df[df[group_by].notnull()]

    # Crear la figura para la gráfica
    fig = go.Figure()

    # Mapeo para convertir 1 y 0 a "con discapacidad" y "sin discapacidad"
    disability_map = {1: "Con discapacidad", 0: "Sin discapacidad"}

    # Graficar para cada grupo de discapacidad (0 y 1)
    for group in df_disability[group_by].unique():
        # Obtener la etiqueta correspondiente (con discapacidad o sin discapacidad)
        disability_label = disability_map.get(group, "Unknown")
        
        # Ajustar el modelo Kaplan-Meier para cada grupo
        kmf.fit(df_disability[df_disability[group_by] == group]['date'],
                event_observed=df_disability[df_disability[group_by] == group]['final_result'],
                label=f"{disability_label}")

        # Obtener el nombre de la columna de probabilidades de supervivencia
        survival_col = kmf.survival_function_.columns[0]

        # Añadir la curva de Kaplan-Meier con bandas de confianza al gráfico
        fig.add_trace(go.Scatter(
            x=kmf.timeline.tolist(),
            y=kmf.survival_function_[survival_col].tolist(),
            mode='lines',
            name=f"{disability_label}",
            line=dict(width=2),
            fill='tonexty',  
            fillcolor='rgba(0, 123, 255, 0.2)' if disability_label == 'Con discapacidad' else 'rgba(255, 165, 0, 0.2)', 
            hovertemplate="(%{x:.0f}, %{y:.3f})"    
        ))


    # Configurar la gráfica
    fig.update_layout(
        title=f'Curva de Kaplan-Meier para la covariable {group_by}',
        xaxis_title='Tiempo',
        yaxis_title='Probabilidad de Supervivencia',
        yaxis=dict(range=[0, 1]),
        legend_title=group_by,
        template='plotly_white', 
    )

    return dcc.Graph(figure=fig)  