from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from dash import dash_table
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

def run_cox_regression(df_limpio, covariables):
    """
    Ejecuta regresión de Cox con las covariables seleccionadas.
    Maneja tanto parámetros simples como compuestos (age_band, highest_education).
    Remueve una columna de referencia para cada variable categórica (one-hot encoded).
    
    Args:
        df_limpio: DataFrame con los datos
        covariables: Lista de nombres de covariables (ej: ['gender_F', 'age_band', 'studied_credits'])
    
    Returns:
        summary: DataFrame con el resumen del modelo
        cox_table_html: Tabla HTML para mostrar
        
    Raises:
        ValueError: Si los datos no son válidos
    """
    
    try:
        # VALIDACIÓN 0: Verificar inputs básicos
        if df_limpio is None or len(df_limpio) == 0:
            raise ValueError("❌ DataFrame vacío - No hay datos para regresión de Cox")
        
        if not covariables or len(covariables) == 0:
            raise ValueError("❌ No se seleccionó ninguna covariable")
        
        # VALIDACIÓN 1: Verificar columnas críticas
        columnas_criticas = ['date', 'final_result']
        faltantes = [col for col in columnas_criticas if col not in df_limpio.columns]
        if faltantes:
            raise ValueError(f"❌ Falta columnas críticas: {', '.join(faltantes)}")
        
        # Suprimir warnings de lifelines sobre convergencia
        import warnings
        warnings.filterwarnings('ignore', category=Warning)
        
        # Expandir los parámetros compuestos a sus columnas reales
        columnas_finales = ['date', 'final_result']
        for cov in covariables:
            if cov in COVARIABLE_MAPPING:
                columnas_finales.extend(COVARIABLE_MAPPING[cov])
            else:
                columnas_finales.append(cov)
        
        # Remover duplicados mientras mantenemos el orden
        columnas_finales = list(dict.fromkeys(columnas_finales))
        
        # Verificar que todas las columnas existan en el DataFrame
        columnas_validas = [col for col in columnas_finales if col in df_limpio.columns]
        
        print(f"[DEBUG Cox] Covariables solicitadas: {covariables}")
        print(f"[DEBUG Cox] Columnas válidas encontradas: {columnas_validas}")
        
        # Preparamos el DataFrame para la regresión de Cox
        df_cox = df_limpio[columnas_validas].copy()
        
        # Remover filas con valores faltantes
        df_cox = df_cox.dropna()
        
        if df_cox.empty:
            print("[DEBUG Cox] DataFrame vacío después de dropna()")
            return pd.DataFrame(), None
        
        # Para variables one-hot encoded (categóricas), remover la última columna para evitar colinealidad perfecta
        cov_cols = [col for col in columnas_validas if col not in ['date', 'final_result']]
        
        # Identificar variables categóricas one-hot encoded y remover la última columna de cada grupo
        columnas_por_grupo = {}
        for col in cov_cols:
            if '_' in col:
                partes = col.rsplit('_', 1)
                if len(partes) == 2:
                    prefijo = partes[0]
                    if prefijo not in columnas_por_grupo:
                        columnas_por_grupo[prefijo] = []
                    columnas_por_grupo[prefijo].append(col)
                else:
                    if 'simple' not in columnas_por_grupo:
                        columnas_por_grupo['simple'] = []
                    columnas_por_grupo['simple'].append(col)
            else:
                if 'simple' not in columnas_por_grupo:
                    columnas_por_grupo['simple'] = []
                columnas_por_grupo['simple'].append(col)
        
        # Remover la última columna de cada grupo categórico (es la referencia)
        cov_cols_ajustadas = []
        for grupo, cols in columnas_por_grupo.items():
            if grupo == 'simple' or len(cols) == 1:
                cov_cols_ajustadas.extend(cols)
            else:
                cols_sin_referencia = cols[:-1]
                cov_cols_ajustadas.extend(cols_sin_referencia)
                print(f"[DEBUG Cox] Variable categórica '{grupo}': referencia: {cols[-1]}")
        
        print(f"[DEBUG Cox] Covariables definitivas: {cov_cols_ajustadas}")
        
        # Preparar DataFrame final
        df_cox = df_cox[['date', 'final_result'] + cov_cols_ajustadas].copy()
        df_cox = df_cox.dropna()
        
        if df_cox.empty:
            print("[DEBUG Cox] DataFrame vacío")
            return pd.DataFrame(), None
        
        # Creamos el objeto CoxPHFitter y ajustamos el modelo
        cph = CoxPHFitter()
        
        # Intentar ajustar con todas las variables
        try:
            cph.fit(df_cox, duration_col='date', event_col='final_result', show_progress=False)
            print("[DEBUG Cox] ✓ Modelo ajustado correctamente")
        except Exception as e:
            print(f"[DEBUG Cox] Error inicial: {str(e)}")
            print(f"[DEBUG Cox] Intentando con regularización...")
            
            fitted_with_fallback = False
            try:
                # Intentar con regularización (penalización)
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(df_cox, duration_col='date', event_col='final_result', show_progress=False)
                fitted_with_fallback = True
                print("[DEBUG Cox] ✓ Modelo ajustado con regularización")
            except Exception as e2:
                print(f"[DEBUG Cox] Error con regularización: {str(e2)}")
                print(f"[DEBUG Cox] Removiendo variables problemáticas...")
            
            # Remover variables problemáticas conocidas una por una
            variables_problematicas = [
                'age_band_55<=',
                'disability_N',
                'highest_education_Lower Than A Level',
                'studied_credits'
            ]
            
            exito = False
            for var_problematica in ([] if fitted_with_fallback else variables_problematicas):
                if var_problematica in cov_cols_ajustadas:
                    cov_cols_temp = [c for c in cov_cols_ajustadas if c != var_problematica]
                    if not cov_cols_temp:
                        continue
                    
                    df_cox_temp = df_cox[['date', 'final_result'] + cov_cols_temp].copy()
                    df_cox_temp = df_cox_temp.dropna()
                    
                    try:
                        cph.fit(df_cox_temp, duration_col='date', event_col='final_result', show_progress=False)
                        print(f"[DEBUG Cox] ✓ Modelo ajustado sin: {var_problematica}")
                        df_cox = df_cox_temp
                        cov_cols_ajustadas = cov_cols_temp
                        exito = True
                        break
                    except:
                        continue
            
            if not fitted_with_fallback and not exito:
                print("[ERROR Cox] No se pudo ajustar el modelo con ninguna combinación de variables")
                # Retornar error pero con información
                error_summary = pd.DataFrame({
                    'Covariable': ['ERROR'],
                    'Estado': ['Modelo de Cox no convergió. Intenta con más variables o más datos.']
                })
                return error_summary, None

        # Resumen del modelo de Cox
        summary = cph.summary.copy()
        
        # ✅ ERROR #4: Validar que tenemos suficientes eventos para el análisis
        total_events = int(df_cox['final_result'].sum())
        min_events = 5
        if total_events < min_events:
            print(f"[WARN Cox] Solo {total_events} eventos observados (mínimo recomendado: {min_events})")
            # Continuar pero con advertencia
        
        # ✅ ERROR #4: Validar que no hay problemas con división por cero en el cálculo de p-valores
        if summary.empty:
            return pd.DataFrame(), None
        
        # Manejo seguro de p-valores que pueden ser muy pequeños o indefinidos
        summary['p'] = summary['p'].fillna(1.0)  # Reemplazar NaN con 1.0 (no significativo)
        summary['-log2(p)'] = -np.log2(summary['p'].clip(lower=1e-10))  # Evitar log de 0

        # Tabla HTML a partir del resumen del modelo de Cox
        cols_mostrar = ['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%',
                        'exp(coef) lower 95%', 'exp(coef) upper 95%', 'cmp to', 'z', 'p', '-log2(p)']
        summary = summary[[c for c in cols_mostrar if c in summary.columns]]
        
        summary.reset_index(inplace=True)
        summary.columns = ['Covariable', 'Coef.', 'exp(Coef.)', 'SE(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%',
                           'exp(Coef.) lower 95%', 'exp(Coef.) upper 95%', 'cmp to', 'z', 'p', '-log2(p)']

        # Crear el DataTable
        cox_table_html = dash_table.DataTable(
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
            style_cell_conditional=[
                {'if': {'column_id': 'Covariable'}, 'textAlign': 'center'}
            ]
        )

        # Retornar tanto el resumen como la tabla HTML
        return summary, cox_table_html
    
    except Exception as e:
        print(f"[ERROR Cox] {str(e)}")
        return pd.DataFrame(), None


def create_forest_plot(summary_data):
    """
    Crea un Forest Plot para visualizar los coeficientes de Cox Regression.
    
    Args:
        summary_data: DataFrame con el resumen del modelo Cox
                      Esperado: columnas 'Covariable', 'exp(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%'
    
    Returns:
        go.Figure: Figura Plotly del Forest Plot, o None si hay error
    """
    try:
        if summary_data is None or summary_data.empty:
            print("[FOREST] No hay datos para crear Forest Plot")
            return None
        
        # Verificar columnas necesarias
        cols_requeridas = ['Covariable', 'exp(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%']
        if not all(col in summary_data.columns for col in cols_requeridas):
            print(f"[FOREST] Columnas faltantes. Disponibles: {list(summary_data.columns)}")
            return None
        
        # Preparar datos y usar HR con intervalos coherentes
        summary_data = summary_data.copy()
        summary_data['HR'] = pd.to_numeric(summary_data['exp(Coef.)'], errors='coerce')
        summary_data['CI_lower'] = pd.to_numeric(summary_data['exp(Coef.) lower 95%'], errors='coerce') if 'exp(Coef.) lower 95%' in summary_data.columns else np.exp(pd.to_numeric(summary_data['Coef. lower 95%'], errors='coerce'))
        summary_data['CI_upper'] = pd.to_numeric(summary_data['exp(Coef.) upper 95%'], errors='coerce') if 'exp(Coef.) upper 95%' in summary_data.columns else np.exp(pd.to_numeric(summary_data['Coef. upper 95%'], errors='coerce'))
        summary_data = summary_data.dropna(subset=['HR', 'CI_lower', 'CI_upper'])

        if summary_data.empty:
            print("[FOREST] No quedan filas válidas tras limpiar datos")
            return None

        summary_data = summary_data.iloc[::-1].reset_index(drop=True)
        summary_data['label'] = summary_data['Covariable'].astype(str)
        summary_data['y_pos'] = list(range(len(summary_data)))
        summary_data['log_hr'] = np.log(summary_data['HR'].clip(lower=1e-10))
        summary_data['err_minus'] = summary_data['HR'] - summary_data['CI_lower']
        summary_data['err_plus'] = summary_data['CI_upper'] - summary_data['HR']

        fig = go.Figure()
        fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="No Effect", annotation_position="top")

        fig.add_trace(go.Scatter(
            x=summary_data['HR'],
            y=summary_data['y_pos'],
            mode='markers',
            marker=dict(size=10, color='darkblue', symbol='circle'),
            error_x=dict(
                type='data',
                symmetric=False,
                array=summary_data['err_plus'],
                arrayminus=summary_data['err_minus'],
                color='steelblue',
                thickness=2,
                width=0
            ),
            text=[
                f"HR: {hr:.2f} (IC 95%: {ci_l:.2f}-{ci_u:.2f})"
                for hr, ci_l, ci_u in zip(summary_data['HR'], summary_data['CI_lower'], summary_data['CI_upper'])
            ],
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ))

        fig.update_layout(
            title="Forest Plot - Coeficientes del Modelo Cox",
            xaxis_title="Hazard Ratio",
            yaxis_title="Variables",
            height=max(420, 120 + len(summary_data) * 55),
            hovermode='closest',
            plot_bgcolor='rgba(240,240,240,0.5)',
            margin=dict(l=220, r=80, t=80, b=60),
            yaxis=dict(
                tickmode='array',
                tickvals=summary_data['y_pos'],
                ticktext=summary_data['label'],
                autorange='reversed'
            ),
            xaxis=dict(type='log')
        )
        
        print(f"[FOREST] ✓ Forest Plot creado con {len(summary_data)} variables")
        return fig
        
    except Exception as e:
        print(f"[FOREST] ✗ Error creando Forest Plot: {e}")
        import traceback
        traceback.print_exc()
        return None
