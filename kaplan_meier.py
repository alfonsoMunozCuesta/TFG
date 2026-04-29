from dash import dcc
import plotly.graph_objs as go
from lifelines import KaplanMeierFitter
import pandas as pd

# Mapeo de parámetros a columnas reales
COVARIABLE_MAPPING = {
    'gender_F': ['gender_F'],
    'disability_N': ['disability_N'],
    'age_band': ['age_band_0-35', 'age_band_35-55', 'age_band_55<='],
    'highest_education': ['highest_education_A Level or Equivalent', 
                         'highest_education_HE Qualification',
                         'highest_education_Lower Than A Level', 
                         'highest_education_Post Graduate Qualification'],
    'studied_credits': ['studied_credits']
}


def _rgba_with_alpha(color, alpha):
    if color.startswith('rgba(') and color.endswith(')'):
        parts = [part.strip() for part in color[5:-1].split(',')]
        if len(parts) >= 3:
            return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha})"
    return color


def _add_km_trace_with_ci(fig, kmf, name, color, show_legend=True):
    survival_col = kmf.survival_function_.columns[0]
    timeline = kmf.timeline.tolist()
    survival_values = kmf.survival_function_[survival_col].tolist()
    ci = kmf.confidence_interval_survival_function_
    upper_values = ci.iloc[:, 1].tolist()
    lower_values = ci.iloc[:, 0].tolist()

    fig.add_trace(go.Scatter(
        x=timeline,
        y=upper_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        hoverinfo='skip',
        showlegend=False,
        name=f'{name} IC superior'
    ))

    fig.add_trace(go.Scatter(
        x=timeline,
        y=lower_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillcolor=_rgba_with_alpha(color, 0.18),
        hoverinfo='skip',
        showlegend=False,
        name=f'{name} IC inferior'
    ))

    fig.add_trace(go.Scatter(
        x=timeline,
        y=survival_values,
        mode='lines',
        name=name,
        line=dict(color=_rgba_with_alpha(color, 1.0), width=2),
        hovertemplate='(%{x:.0f}, %{y:.3f})<extra></extra>',
        showlegend=show_legend
    ))

def plot_kaplan_meier(df):
    # ✅ ERROR #3: Validar estructura de datos antes de análisis
    if df is None or len(df) == 0:
        raise ValueError("❌ DataFrame vacío - No hay datos para graficar")
    
    if 'date' not in df.columns or 'final_result' not in df.columns:
        raise ValueError("❌ Faltan columnas críticas: 'date' o 'final_result'")
    
    kmf = KaplanMeierFitter()

    kmf.fit(df['date'], event_observed=df['final_result'])

    # Crear la figura con Plotly
    fig = go.Figure()

    # curva de Kaplan-Meier
    _add_km_trace_with_ci(fig, kmf, '', 'rgba(0, 0, 255, 1.0)', show_legend=False)

    fig.update_layout(
        title="Curva de Supervivencia Kaplan-Meier",
        xaxis_title="Tiempo",
        yaxis_title="Probabilidad de Supervivencia",
        yaxis=dict(range=[0, 1]),

    )

    return dcc.Graph(figure=fig)


def plot_km_G(df, group_by='gender_F'):
    """Función genérica para graficar Kaplan-Meier por cualquier covariable"""
    return plot_km_by_covariate(df, group_by)

def plot_km_disc(df, group_by='disability_N'):
    """Función genérica para graficar Kaplan-Meier por discapacidad"""
    return plot_km_by_covariate(df, group_by)

def plot_km_by_covariate(df, covariable_name='gender_F'):
    """
    Función genérica para graficar Kaplan-Meier agrupado por cualquier covariable.
    Maneja tanto variables binarias como categóricas con múltiples grupos.
    
    Raises:
        ValueError: Si faltan datos críticos en el dataframe
    """
    # VALIDACIÓN 1: Verificar que el dataframe no está vacío
    if df is None or len(df) == 0:
        raise ValueError("❌ DataFrame vacío - No hay datos para graficar")
    
    # VALIDACIÓN 2: Verificar que existen columnas críticas
    columnas_criticas = ['date', 'final_result']
    faltantes = [col for col in columnas_criticas if col not in df.columns]
    if faltantes:
        raise ValueError(f"❌ Falta columnas críticas: {', '.join(faltantes)}")
    
    # VALIDACIÓN 3: Verificar que la covariable solicitada existe
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    cols_disponibles = [col for col in columns if col in df.columns]
    if not cols_disponibles:
        raise ValueError(f"❌ La covariable '{covariable_name}' no existe en el dataset")
    
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    
    # Obtener columnas asociadas a la covariable
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    
    # Mapeo de etiquetas según el tipo de variable
    label_map = {
        'gender_F': {1: "Femenino", 0: "Masculino"},
        'disability_N': {1: "Con discapacidad", 0: "Sin discapacidad"},
        'studied_credits': {}  # Sin mapeo para variable continua
    }
    label_mapping = label_map.get(covariable_name, {})
    
    colors = ['rgba(0, 123, 255, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(0, 200, 0, 0.3)', 
             'rgba(200, 0, 0, 0.3)', 'rgba(128, 0, 128, 0.3)', 'rgba(255, 192, 203, 0.3)']
    
    # Para variables binarias simples (una columna, excepto studied_credits)
    if len(columns) == 1 and covariable_name != 'studied_credits':
        col = columns[0]
        df_filtered = df[df[col].notnull()]
        
        # Graficar para cada grupo
        groups = sorted(df_filtered[col].unique())
        for idx, group in enumerate(groups):
            df_group = df_filtered[df_filtered[col] == group]
            
            if len(df_group) > 0:
                # Obtener etiqueta
                if label_mapping:
                    group_label = label_mapping.get(group, str(group))
                else:
                    group_label = f"{col} = {group}"
                
                # Ajustar Kaplan-Meier
                kmf.fit(df_group['date'], 
                       event_observed=df_group['final_result'],
                       label=group_label)

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    group_label,
                    colors[idx % len(colors)]
                )
    
    # Para variables categóricas multi-valor (age_band, highest_education)
    elif len(columns) > 1:
        idx = 0
        # Iterar sobre TODAS las columnas mapeadas, incluso si no existen
        for col in columns:
            # Extraer label
            parts = col.split('_', 1)
            if len(parts) > 1:
                label = parts[1]
            else:
                label = col
            
            # Obtener datos si existen
            if col in df.columns:
                df_group = df[df[col] == 1].copy()
            else:
                df_group = pd.DataFrame()  # Columna no existe
            
            if len(df_group) > 0:
                # Ajustar Kaplan-Meier
                kmf.fit(df_group['date'],
                       event_observed=df_group['final_result'],
                       label=f"{label} (n={len(df_group)})")

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    f"{label} (n={len(df_group)})",
                    colors[idx % len(colors)]
                )
            else:
                # Grupo sin observaciones - mostrar en leyenda como línea punteada gris
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines+markers',
                    name=f"{label} (n=0 - Sin datos)",
                    line=dict(width=2, dash='dash', color='lightgray'),
                    marker=dict(symbol='x'),
                    hovertemplate=f"<b>{label}</b><br>Sin observaciones en este grupo<extra></extra>"
                ))
            idx += 1
    
    # Para studied_credits (variable continua/numérica)
    elif covariable_name == 'studied_credits':
        col = columns[0]
        df_filtered = df[df[col].notnull()].copy()
        
        if len(df_filtered) > 0:
            # Obtener valores únicos y ordenarlos
            unique_values = sorted(df_filtered[col].unique())
            
            # Si hay demasiados valores, agrupar en rangos
            if len(unique_values) > 10:
                # Dividir en cuartiles o rangos
                df_filtered['credits_group'] = pd.cut(df_filtered[col], bins=5, duplicates='drop')
                groups_col = 'credits_group'
            else:
                df_filtered['credits_group'] = df_filtered[col]
                groups_col = 'credits_group'
            
            for idx, group_val in enumerate(df_filtered[groups_col].unique()):
                df_group = df_filtered[df_filtered[groups_col] == group_val]
                
                if len(df_group) > 0:
                    group_label = f"Créditos: {group_val}"
                    
                    kmf.fit(df_group['date'],
                           event_observed=df_group['final_result'],
                           label=group_label)

                    _add_km_trace_with_ci(
                        fig,
                        kmf,
                        group_label,
                        colors[idx % len(colors)]
                    )
    
    # Configurar la gráfica
    fig.update_layout(
        title=f'Curva de Kaplan-Meier para {covariable_name}',
        xaxis_title='Tiempo',
        yaxis_title='Probabilidad de Supervivencia',
        yaxis=dict(range=[0, 1]),
        legend_title=covariable_name,
        template='plotly_white',
    )
    
    return dcc.Graph(figure=fig)


def plot_km_by_covariate_with_figure(df, covariable_name='gender_F'):
    """
    Retorna TANTO el componente dcc.Graph COMO la figura JSON para guardar en Store.
    
    Returns:
        Tuple[dcc.Graph, dict]: (componente gráfico, figura JSON)
    """
    # Usar la lógica existente para generar la figura
    fig = _create_km_figure(df, covariable_name)
    
    # Retornar el componente gráfico Y la figura JSON
    return dcc.Graph(figure=fig), fig.to_dict()


def _create_km_figure(df, covariable_name='gender_F'):
    """
    Crea y retorna una figura Plotly go.Figure (no dcc.Graph) 
    para poder usarla en múltiples contextos.
    """
    # VALIDACIÓN 1: Verificar que el dataframe no está vacío
    if df is None or len(df) == 0:
        raise ValueError("❌ DataFrame vacío - No hay datos para graficar")
    
    # VALIDACIÓN 2: Verificar que existen columnas críticas
    columnas_criticas = ['date', 'final_result']
    faltantes = [col for col in columnas_criticas if col not in df.columns]
    if faltantes:
        raise ValueError(f"❌ Falta columnas críticas: {', '.join(faltantes)}")
    
    # VALIDACIÓN 3: Verificar que la covariable solicitada existe
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    cols_disponibles = [col for col in columns if col in df.columns]
    if not cols_disponibles:
        raise ValueError(f"❌ La covariable '{covariable_name}' no existe en el dataset")
    
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    
    # Obtener columnas asociadas a la covariable
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    
    # Mapeo de etiquetas según el tipo de variable
    label_map = {
        'gender_F': {1: "Femenino", 0: "Masculino"},
        'disability_N': {1: "Con discapacidad", 0: "Sin discapacidad"},
        'studied_credits': {}  # Sin mapeo para variable continua
    }
    label_mapping = label_map.get(covariable_name, {})
    
    colors = ['rgba(0, 123, 255, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(0, 200, 0, 0.3)', 
             'rgba(200, 0, 0, 0.3)', 'rgba(128, 0, 128, 0.3)', 'rgba(255, 192, 203, 0.3)']
    
    # Para variables binarias simples (una columna, excepto studied_credits)
    if len(columns) == 1 and covariable_name != 'studied_credits':
        col = columns[0]
        df_filtered = df[df[col].notnull()]
        
        # Graficar para cada grupo
        groups = sorted(df_filtered[col].unique())
        for idx, group in enumerate(groups):
            df_group = df_filtered[df_filtered[col] == group]
            
            if len(df_group) > 0:
                # Obtener etiqueta
                if label_mapping:
                    group_label = label_mapping.get(group, str(group))
                else:
                    group_label = f"{col} = {group}"
                
                # Ajustar Kaplan-Meier
                kmf.fit(df_group['date'], 
                       event_observed=df_group['final_result'],
                       label=group_label)

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    group_label,
                    colors[idx % len(colors)]
                )
    
    # Para variables categóricas multi-valor (age_band, highest_education)
    elif len(columns) > 1:
        idx = 0
        # Procesar solo las columnas que existen en el dataframe
        for col in columns:
            # Solo procesar si la columna existe
            if col not in df.columns:
                continue
            
            # Extraer label de la columna
            parts = col.split('_', 1)
            if len(parts) > 1:
                label = parts[1]
            else:
                label = col
            
            # Obtener registros donde esta columna vale 1
            df_group = df[df[col] == 1].copy()
            
            if len(df_group) > 0:
                kmf.fit(df_group['date'],
                       event_observed=df_group['final_result'],
                       label=label)

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    label,
                    colors[idx % len(colors)]
                )
                idx += 1
    
    # Para variables continuas (studied_credits)
    else:
        for col in columns:
            if col in df.columns:
                df_filtered = df[[col, 'date', 'final_result']].dropna()
                
                if len(df_filtered) > 0:
                    # Dividir en cuartiles o rangos
                    df_filtered = df_filtered.copy()
                    df_filtered['credits_group'] = pd.cut(df_filtered[col], bins=5, duplicates='drop')
                    groups_col = 'credits_group'
                    
                    for idx, group_val in enumerate(df_filtered[groups_col].unique()):
                        df_group = df_filtered[df_filtered[groups_col] == group_val]
                        
                        if len(df_group) > 0:
                            group_label = f"Créditos: {group_val}"
                            
                            kmf.fit(df_group['date'],
                                   event_observed=df_group['final_result'],
                                   label=group_label)

                            _add_km_trace_with_ci(
                                fig,
                                kmf,
                                group_label,
                                colors[idx % len(colors)]
                            )
    
    # Configurar la gráfica
    fig.update_layout(
        title=f'Curva de Kaplan-Meier para {covariable_name}',
        xaxis_title='Tiempo',
        yaxis_title='Probabilidad de Supervivencia',
        yaxis=dict(range=[0, 1]),
        legend_title=covariable_name,
        template='plotly_white',
    )
    
    return fig
