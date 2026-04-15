"""
Funciones para crear gráficas de análisis de supervivencia.
Incluye gráficas para Log-Rank Test y Cox Regression.
"""

import plotly.graph_objs as go
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from dash import dcc, html
from config import df_limpio

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

# Mapeo de labels legibles
LABEL_MAPPING = {
    'gender_F': 'Género',
    'disability_N': 'Discapacidad',
    'age_band_0-35': '0-35 años',
    'age_band_35-55': '35-55 años',
    'age_band_55<=': '55+ años',
    'highest_education_A Level or Equivalent': 'A Level',
    'highest_education_HE Qualification': 'HE Qualification',
    'highest_education_Lower Than A Level': 'Lower Than A Level',
    'highest_education_Post Graduate Qualification': 'Postgrado',
    'studied_credits': 'Créditos Estudiados'
}


def plot_logrank_curves(df, covariable, language='es'):
    """
    Crea una gráfica de curvas de supervivencia (Kaplan-Meier) estratificadas por una covariable.
    
    Args:
        df: DataFrame con los datos
        covariable: Nombre de la covariable para estratificar
    
    Returns:
        dcc.Graph con la gráfica de Kaplan-Meier estratificada
    """
    
    is_en = language == 'en'
    columns = COVARIABLE_MAPPING.get(covariable, [covariable])
    
    fig = go.Figure()
    kmf = KaplanMeierFitter()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Si es variable continua (studied_credits)
    if covariable == 'studied_credits':
        df_temp = df[['studied_credits', 'date', 'final_result']].dropna()
        
        # Crear quintiles
        n_unique = df_temp['studied_credits'].nunique()
        if n_unique <= 5:
            unique_vals = sorted(df_temp['studied_credits'].unique())
            value_to_group = {val: i for i, val in enumerate(unique_vals)}
            df_temp['group'] = df_temp['studied_credits'].map(value_to_group)
            groups = sorted(df_temp['group'].unique())
            labels = [f"Quintile {i+1}\n({unique_vals[i]:.0f} credits)" if is_en else f"Quintil {i+1}\n({unique_vals[i]:.0f} créditos)" for i in groups]
        else:
            df_temp['group'] = pd.qcut(df_temp['studied_credits'], q=5, labels=False, duplicates='drop')
            groups = sorted(df_temp['group'].unique())
            labels = [f"Quintile {i+1}" if is_en else f"Quintil {i+1}" for i in groups]
        
        for i, group_val in enumerate(groups):
            df_group = df_temp[df_temp['group'] == group_val]
            if len(df_group) > 0:
                kmf.fit(df_group['date'], event_observed=df_group['final_result'], label=labels[i])
                
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=kmf.survival_function_.iloc[:, 0].tolist(),
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                        if is_en else
                        "<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
                    )
                ))
                
                # Añadir intervalos de confianza
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 1].tolist(),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Upper CI' if is_en else 'IC superior'})"
                ))
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 0].tolist(),
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Lower CI' if is_en else 'IC inferior'})",
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.2)"
                ))
    
    # Si es variable categórica multi-clase (age_band, highest_education)
    elif len(columns) > 1:
        # Iterar sobre TODAS las columnas mapeadas, incluso si no existen en el dataframe actual
        groups_info = []
        for col in columns:
            # Extraer label directamente del mapeo
            label = LABEL_MAPPING.get(col, col.split('_', 1)[1] if '_' in col else col)
            
            # Obtener datos si la columna existe, si no, un dataframe vacío
            if col in df.columns:
                df_group = df[df[col] == 1]
            else:
                df_group = pd.DataFrame()  # Columna no existe en datos
            
            groups_info.append({
                'col': col, 
                'label': label, 
                'data': df_group, 
                'n': len(df_group)
            })
        
        print(f'[LOGRANK CURVES] Grupos mapeados: {[info["label"] + f" (n={info["n"]})" for info in groups_info]}')
        
        for i, info in enumerate(groups_info):
            if len(info['data']) > 0:
                kmf.fit(info['data']['date'], event_observed=info['data']['final_result'], 
                        label=f"{info['label']} (n={info['n']})")
                
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=kmf.survival_function_.iloc[:, 0].tolist(),
                    mode='lines',
                    name=f"{info['label']} (n={info['n']})",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                        if is_en else
                        "<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
                    )
                ))
                
                # Intervalos de confianza
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 1].tolist(),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{info['label']} ({'Upper CI' if is_en else 'IC superior'})"
                ))
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 0].tolist(),
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{info['label']} ({'Lower CI' if is_en else 'IC inferior'})",
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.2)"
                ))
            else:
                # Grupo sin datos - mostrar como línea punteada gris con n=0
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines+markers',
                    name=f"{info['label']} (n=0 - {'No data' if is_en else 'Sin datos'})",
                    line=dict(color='lightgray', width=2, dash='dash'),
                    marker=dict(symbol='x'),
                    hovertemplate=f"<b>{info['label']}</b><br>{'No observations in this group' if is_en else 'Sin observaciones en este grupo'}<extra></extra>"
                ))
    
    # Si es variable binaria simple (gender_F, disability_N)
    else:
        col = columns[0]
        groups = sorted(df[col].dropna().unique())
        labels = [f"{LABEL_MAPPING.get(col, col)}: {vals}" for vals in groups]
        
        for i, group_val in enumerate(groups):
            df_group = df[df[col] == group_val]
            if len(df_group) > 0:
                kmf.fit(df_group['date'], event_observed=df_group['final_result'], label=labels[i])
                
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=kmf.survival_function_.iloc[:, 0].tolist(),
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                        if is_en else
                        "<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
                    )
                ))
                
                # Intervalos de confianza
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 1].tolist(),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Upper CI' if is_en else 'IC superior'})"
                ))
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 0].tolist(),
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Lower CI' if is_en else 'IC inferior'})",
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.2)"
                ))
    
    fig.update_layout(
        title=(
            f"Survival Curves (Kaplan-Meier) - {LABEL_MAPPING.get(covariable, covariable)}"
            if is_en else
            f"Curvas de Supervivencia (Kaplan-Meier) - {LABEL_MAPPING.get(covariable, covariable)}"
        ),
        xaxis_title="Time" if is_en else "Tiempo",
        yaxis_title="Survival Probability" if is_en else "Probabilidad de Supervivencia",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return dcc.Graph(figure=fig)


def plot_cox_hazard_ratios(summary_df, covariable_name, language='es'):
    """
    Crea una gráfica de Forest Plot (Hazard Ratios con intervalos de confianza).
    
    Args:
        summary_df: DataFrame con los resultados de Cox Regression (de cph.summary)
        covariable_name: Nombre de la covariable
    
    Returns:
        dcc.Graph con el Forest Plot
    """
    
    # Asegurar que tenemos los datos correctos
    summary_df = summary_df.copy()
    
    # Detectar nombres de columnas (pueden variar entre versiones de lifelines)
    coef_col = 'coef' if 'coef' in summary_df.columns else 'Coef.'
    se_col = 'se(coef)' if 'se(coef)' in summary_df.columns else 'SE(Coef.)'
    
    # Calcular HR (Hazard Ratio) y sus intervalos de confianza
    summary_df['HR'] = np.exp(pd.to_numeric(summary_df[coef_col], errors='coerce'))
    summary_df['se_coef'] = pd.to_numeric(summary_df[se_col], errors='coerce')
    summary_df['HR_lower'] = np.exp(pd.to_numeric(summary_df[coef_col], errors='coerce') - 1.96 * summary_df['se_coef'])
    summary_df['HR_upper'] = np.exp(pd.to_numeric(summary_df[coef_col], errors='coerce') + 1.96 * summary_df['se_coef'])
    
    # Remover filas con NaN
    summary_df = summary_df.dropna(subset=['HR', 'HR_lower', 'HR_upper'])
    
    is_en = language == 'en'

    if summary_df.empty:
        return dcc.Graph(figure=go.Figure().add_annotation(text="Could not compute Hazard Ratios" if is_en else "No se pudieron calcular Hazard Ratios"))
    
    fig = go.Figure()
    
    # Detectar nombre de variablecolumna
    var_col = 'Covariable' if 'Covariable' in summary_df.columns else summary_df.columns[0]
    
    colors = ['green' if hr < 1 else 'red' for hr in summary_df['HR']]
    
    # Añadir línea de referencia vertical en HR=1 (sin efecto)
    fig.add_vline(x=1, line_dash="dash", line_color="black", 
                  annotation_text="HR = 1 (No effect)" if is_en else "HR = 1 (Sin efecto)", annotation_position="top",
                  annotation_textangle=0)
    
    # Añadir los puntos y barras de error
    for idx, (i, row) in enumerate(summary_df.iterrows()):
        var_name = str(row[var_col]) if var_col in row and pd.notna(row[var_col]) else f"Var {idx}"
        
        fig.add_trace(go.Scatter(
            x=[row['HR']],
            y=[idx],
            mode='markers',
            marker=dict(size=12, color=colors[idx]),
            name=var_name,
            hovertemplate=f"<b>{var_name}</b><br>HR: %{{x:.3f}}<br>IC 95%: ({row['HR_lower']:.3f} - {row['HR_upper']:.3f})<extra></extra>"
        ))
        
        # Barras de error (líneas horizontales)
        fig.add_trace(go.Scatter(
            x=[row['HR_lower'], row['HR_upper']],
            y=[idx, idx],
            mode='lines',
            line=dict(color=colors[idx], width=2),
            showlegend=False,
            hoverinfo='none'
        ))
    
    fig.update_layout(
        title=f"Forest Plot - Hazard Ratios (Cox Regression)",
        xaxis_title="Hazard Ratio (log scale)" if is_en else "Hazard Ratio (escala logarítmica)",
        yaxis_title="Variables" if is_en else "Variables",
        yaxis=dict(autorange="reversed", tickmode='linear', tick0=0),
        xaxis=dict(type="log"),
        height=max(400, len(summary_df) * 50),
        template='plotly_white',
        hovermode='closest',
        showlegend=False
    )
    
    return dcc.Graph(figure=fig)
