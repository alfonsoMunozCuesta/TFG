from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

def perform_log_rank_test(df, group_by, alpha: float = 0.05):
    """
    Realiza el test de Log-Rank para comparar curvas de supervivencia.
    Maneja tanto variables binarias como categóricas.
    Para variables continuas (studied_credits), agrupa en quintiles.
    
    Raises:
        ValueError: Si los datos no son válidos
    """
    
    # VALIDACIÓN 0: Verificar inputs básicos
    if df is None or len(df) == 0:
        raise ValueError("❌ DataFrame vacío - No hay datos para Log-Rank test")
    
    if not group_by:
        raise ValueError("❌ No se seleccionó covariable")
    
    # VALIDACIÓN 1: Verificar columnas críticas
    columnas_criticas = ['date', 'final_result']
    faltantes = [col for col in columnas_criticas if col not in df.columns]
    if faltantes:
        raise ValueError(f"❌ Falta columnas críticas: {', '.join(faltantes)}")
    
    print(f"[LOGRANK] Iniciando perform_log_rank_test para: {group_by}")
    print(f"[LOGRANK] Columnas disponibles: {list(df.columns)}")
    
    # Obtener las columnas asociadas a la covariable
    columns = COVARIABLE_MAPPING.get(group_by, [group_by])
    print(f"[LOGRANK] Columnas mapeadas para '{group_by}': {columns}")
    
    # Si es una variable con múltiples columnas one-hot (age_band, highest_education)
    if len(columns) > 1:
        print(f"[LOGRANK] Es variable categórica multi-grupo")
        return _perform_log_rank_multicategory(df, group_by, columns, alpha)
    
    col = columns[0]
    print(f"[LOGRANK] Columna a procesar: {col}")
    
    # Para studied_credits (variable continua), agrupar en quintiles
    if group_by == 'studied_credits':
        print(f"[LOGRANK] Detectada variable continua: studied_credits - crear quintiles")
        return _perform_log_rank_quintiles(df, group_by, col, alpha)
    
    # Si es una variable binaria o numérica
    vals = sorted(pd.unique(df[col].dropna()))
    
    if len(vals) != 2:
        # Para variables continuas o con más de 2 valores
        if len(vals) > 2:
            return _perform_log_rank_multicategory(df, group_by, columns, alpha)
        else:
            return pd.DataFrame({
                'Covariable': [group_by],
                'Grupo A': [vals[0] if len(vals) > 0 else None],
                'Grupo B': [vals[1] if len(vals) > 1 else None],
                'n A': [np.nan], 'n B': [np.nan],
                'test_statistic': [np.nan], 'p_value': [np.nan], '-log2(p)': [np.nan],
                'Decisión': ['No evaluable'],
                'Conclusión': ['La covariable no tiene exactamente dos grupos.']
            })
    
    # Test de Log-Rank para variables binarias
    a, b = vals[0], vals[1]
    gA = df[df[col] == a]
    gB = df[df[col] == b]

    res = logrank_test(
        gA['date'], gB['date'],
        event_observed_A=gA['final_result'],
        event_observed_B=gB['final_result']
    )

    p = float(res.p_value)
    stat = float(res.test_statistic)
    decision = 'Rechazar H0' if p < alpha else 'No rechazar H0'
    interpretacion = ('Hay diferencias significativas entre las curvas'
                      if p < alpha else
                      'No se observan diferencias significativas entre las curvas')

    return pd.DataFrame({
        'Covariable': [group_by],
        'Grupo A': [a],
        'Grupo B': [b],
        'n A': [int(len(gA))], 'n B': [int(len(gB))],
        'test_statistic': [stat], 'p_value': [p],
        '-log2(p)': [(-np.log2(p) if p > 0 else np.inf)],
        'Decisión': [decision],
        'Conclusión': [interpretacion],
    })


def _perform_log_rank_quintiles(df, group_by, col, alpha: float = 0.05):
    """
    Agrupa una variable continua en quintiles (o grupos por valores únicos si hay pocos).
    """
    print(f"\n[LOGRANK QUINTILES] Iniciando para {group_by}, columna: {col}")
    
    results = []
    
    # Crear quintiles
    df_temp = df[[col, 'date', 'final_result']].copy()
    df_temp = df_temp.dropna()
    
    print(f"[LOGRANK QUINTILES] Rows después de dropna: {len(df_temp)}")
    
    if len(df_temp) == 0:
        print(f"[LOGRANK QUINTILES] ERROR: DataFrame vacío después de dropna")
        return pd.DataFrame()
    
    # Verificar variabilidad en los datos
    n_unique = df_temp[col].nunique()
    print(f"[LOGRANK QUINTILES] Valores únicos en {col}: {n_unique}")
    
    if n_unique < 2:
        print(f"[LOGRANK QUINTILES] ERROR: La variable no tiene suficiente variabilidad")
        return pd.DataFrame()
    
    # ESTRATEGIA: Si hay pocos valores únicos (<=5), usarlos directamente como grupos
    # Si hay muchos, usar qcut() con percentiles
    if n_unique <= 5:
        print(f"[LOGRANK QUINTILES] Usando estrategia: Grupos por valores únicos ({n_unique} grupos)")
        # Asignar cada valor único a un grupo
        unique_vals = sorted(df_temp[col].unique())
        value_to_group = {val: i for i, val in enumerate(unique_vals)}
        df_temp['quintil'] = df_temp[col].map(value_to_group)
        actual_bins = len(df_temp['quintil'].unique())
        print(f"[LOGRANK QUINTILES] ✓ Creados {actual_bins} grupos por valores únicos")
    else:
        print(f"[LOGRANK QUINTILES] Usando estrategia: Percentiles (qcut)")
        # Intentar crear quintiles, reduciendo si es necesario
        for n_bins in [5, 4, 3, 2]:
            try:
                df_temp['quintil'] = pd.qcut(df_temp[col], q=n_bins, labels=False, duplicates='drop')
                actual_bins = len(df_temp['quintil'].unique())
                print(f"[LOGRANK QUINTILES] ✓ Creados {actual_bins} grupos (solicitados {n_bins})")
                break
            except Exception as e:
                if n_bins == 2:
                    print(f"[LOGRANK QUINTILES] ERROR: No se pueden crear ni 2 grupos")
                    return pd.DataFrame()
                continue
    
    # Obtener los quintiles únicos y ordenarlos correctamente
    quintiles = sorted(df_temp['quintil'].unique())
    
    print(f"[LOGRANK QUINTILES] Quintiles encontrados: {quintiles}")
    print(f"[LOGRANK QUINTILES] Número de bins: {len(quintiles)}")
    
    # Si solo se generó 1 bin, no hay comparaciones posibles
    if len(quintiles) <= 1:
        print(f"[LOGRANK QUINTILES] ERROR: Solo se creó 1 grupo")
        return pd.DataFrame()
    
    # Hacer comparaciones pairwise entre quintiles
    for i in range(len(quintiles)):
        for j in range(i + 1, len(quintiles)):
            q_a = quintiles[i]
            q_b = quintiles[j]
            
            gA = df_temp[df_temp['quintil'] == q_a]
            gB = df_temp[df_temp['quintil'] == q_b]
            
            print(f"[LOGRANK QUINTILES] Comparando Quintil {q_a+1} (n={len(gA)}) vs Quintil {q_b+1} (n={len(gB)})")
            
            if len(gA) > 0 and len(gB) > 0:
                try:
                    res = logrank_test(
                        gA['date'], gB['date'],
                        event_observed_A=gA['final_result'],
                        event_observed_B=gB['final_result']
                    )
                    
                    p = float(res.p_value)
                    stat = float(res.test_statistic)
                    decision = 'Rechazar H0' if p < alpha else 'No rechazar H0'
                    interpretacion = ('Hay diferencias significativas' if p < alpha 
                                    else 'No hay diferencias significativas')
                    
                    results.append({
                        'Covariable': f"{group_by}: Quintil {q_a+1} vs Quintil {q_b+1}",
                        'Grupo A': f"Quintil {q_a+1}",
                        'Grupo B': f"Quintil {q_b+1}",
                        'n A': int(len(gA)),
                        'n B': int(len(gB)),
                        'test_statistic': stat,
                        'p_value': p,
                        '-log2(p)': (-np.log2(p) if p > 0 else np.inf),
                        'Decisión': decision,
                        'Conclusión': interpretacion,
                    })
                    print(f"[LOGRANK QUINTILES] ✓ Test completado para Quintil {q_a+1} vs Quintil {q_b+1}")
                except Exception as e:
                    print(f"[LOGRANK QUINTILES] ERROR en test para Quintil {q_a+1} vs Quintil {q_b+1}: {str(e)}")
    
    result_df = pd.DataFrame(results)
    print(f"[LOGRANK QUINTILES] Total de resultados: {len(result_df)}")
    return result_df


def _perform_log_rank_multicategory(df, group_by, columns, alpha: float = 0.05):
    """
    Realiza comparaciones pairwise de Log-Rank para variables categóricas con >2 grupos.
    Muestra TODOS los grupos, incluso si están vacíos.
    """
    results = []
    
    # Encontrar TODOS los grupos del mapeo, incluso si no existen en el dataframe actual
    groups_info = []
    for col in columns:
        # Extraer etiqueta del nombre de columna
        label = col.split('_', 1)[1] if '_' in col else col
        
        # Obtener datos si la columna existe
        if col in df.columns:
            df_group = df[df[col] == 1]
        else:
            # Columna no existe - crear dataframe vacío
            df_group = pd.DataFrame()
        
        groups_info.append({
            'column': col,
            'label': label,
            'data': df_group,
            'n': len(df_group)
        })
    
    print(f"[LOGRANK MULTICATEGORY] Grupos mapeados para {group_by}: {len(groups_info)}")
    for info in groups_info:
        print(f"  - {info['label']}: {info['n']} observaciones")
    
    # Hacer comparaciones pairwise entre grupos
    for i in range(len(groups_info)):
        for j in range(i + 1, len(groups_info)):
            gA = groups_info[i]['data']
            gB = groups_info[j]['data']
            label_a = groups_info[i]['label']
            label_b = groups_info[j]['label']
            n_a = len(gA)
            n_b = len(gB)
            
            if n_a > 0 and n_b > 0:
                try:
                    res = logrank_test(
                        gA['date'], gB['date'],
                        event_observed_A=gA['final_result'],
                        event_observed_B=gB['final_result']
                    )
                    
                    p = float(res.p_value)
                    stat = float(res.test_statistic)
                    decision = 'Rechazar H0' if p < alpha else 'No rechazar H0'
                    interpretacion = ('Hay diferencias significativas' if p < alpha 
                                    else 'No hay diferencias significativas')
                    
                    results.append({
                        'Covariable': f"{group_by}: {label_a} vs {label_b}",
                        'Grupo A': label_a,
                        'Grupo B': label_b,
                        'n A': int(n_a),
                        'n B': int(n_b),
                        'test_statistic': stat,
                        'p_value': p,
                        '-log2(p)': (-np.log2(p) if p > 0 else np.inf),
                        'Decisión': decision,
                        'Conclusión': interpretacion,
                    })
                except Exception as e:
                    print(f"[LOGRANK MULTICATEGORY] Error en comparación: {str(e)}")
            else:
                # Al menos uno está vacío - registrar como no evaluable
                grupo_vacio = label_a if n_a == 0 else label_b
                results.append({
                    'Covariable': f"{group_by}: {label_a} vs {label_b}",
                    'Grupo A': label_a,
                    'Grupo B': label_b,
                    'n A': int(n_a),
                    'n B': int(n_b),
                    'test_statistic': np.nan,
                    'p_value': np.nan,
                    '-log2(p)': np.nan,
                    'Decisión': 'No evaluable',
                    'Conclusión': f'Grupo vacío: {grupo_vacio} (n=0)',
                })
    
    return pd.DataFrame(results)


def create_logrank_figure(df, group_by):
    """
    Crea una gráfica Plotly con las curvas Kaplan-Meier para comparación Log-Rank.
    
    Args:
        df: DataFrame con columnas 'date', 'final_result' y la covariable
        group_by: Nombre de la covariable para agrupar
    
    Returns:
        go.Figure: Figura Plotly con curvas de supervivencia
    """
    try:
        if df is None or df.empty:
            print("[LOGRANK FIGURE] Error: DataFrame vacío")
            return None
        
        # Obtener las columnas asociadas a la covariable
        columns = COVARIABLE_MAPPING.get(group_by, [group_by])
        columns_presentes = [col for col in columns if col in df.columns]

        # Preparar datos
        df_clean = df[['date', 'final_result'] + columns_presentes].dropna(subset=['date', 'final_result']) if columns_presentes else df[['date', 'final_result']].dropna(subset=['date', 'final_result'])

        # Crear figura
        fig = go.Figure()
        kmf = KaplanMeierFitter()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Variable continua: studied_credits
        if group_by == 'studied_credits' and 'studied_credits' in df.columns:
            df_temp = df[['studied_credits', 'date', 'final_result']].dropna()
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
                if len(group_df) > 0:
                    kmf.fit(group_df['date'], event_observed=group_df['final_result'], label=labels[i])
                    fig.add_trace(go.Scatter(
                        x=kmf.timeline.tolist(),
                        y=kmf.survival_function_.iloc[:, 0].tolist(),
                        mode='lines',
                        name=labels[i],
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>'
                    ))

        # Variables multi-categoría one-hot (age_band, highest_education)
        elif len(columns_presentes) > 1:
            for i, col in enumerate(columns_presentes):
                label = col.split('_', 1)[1] if '_' in col else col
                group_df = df[df[col] == 1]
                if len(group_df) > 0:
                    kmf.fit(group_df['date'], event_observed=group_df['final_result'], label=f"{label} (n={len(group_df)})")
                    fig.add_trace(go.Scatter(
                        x=kmf.timeline.tolist(),
                        y=kmf.survival_function_.iloc[:, 0].tolist(),
                        mode='lines',
                        name=f"{label} (n={len(group_df)})",
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        name=f"{label} (n=0)",
                        line=dict(color='lightgray', width=2, dash='dash'),
                        showlegend=True
                    ))

        # Variables binarias simples
        elif len(columns_presentes) == 1:
            col = columns_presentes[0]
            groups = sorted(df_clean[col].dropna().unique()) if col in df_clean.columns else []

            for i, group_val in enumerate(groups):
                group_df = df_clean[df_clean[col] == group_val]
                if len(group_df) > 0:
                    kmf.fit(group_df['date'], event_observed=group_df['final_result'], label=f"{group_by}={group_val}")
                    fig.add_trace(go.Scatter(
                        x=kmf.timeline.tolist(),
                        y=kmf.survival_function_.iloc[:, 0].tolist(),
                        mode='lines',
                        name=f"{group_val} (n={len(group_df)})",
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>'
                    ))

        # Configurar layout
        fig.update_layout(
            title=f"Curvas Kaplan-Meier - {group_by}",
            xaxis_title="Tiempo (meses)",
            yaxis_title="Función de Supervivencia",
            hovermode='x unified',
            plot_bgcolor='rgba(240,240,240,0.5)',
            width=1200,
            height=600,
            font=dict(size=12),
            yaxis=dict(range=[0, 1])
        )
        
        print(f"[LOGRANK FIGURE] ✓ Gráfica creada para {group_by}")
        return fig
        
    except Exception as e:
        print(f"[LOGRANK FIGURE] ✗ Error creando gráfica: {e}")
        import traceback
        traceback.print_exc()
        return None
