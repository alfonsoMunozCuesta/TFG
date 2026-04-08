import pandas as pd
import numpy as np 
import matplotlib.pyplot as pyplot
import os

def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=';')
    df.columns = df.columns.str.strip()  # Elimina espacios en los nombres de las columnas
    return df

# Función para limpiar las columnas de actividad
def clean_columns(df):
    columnas_actividad = [
        "ouelluminate", "sharedsubpage", "ouwiki", "folder", "page", "externalquiz",
        "quiz", "dualpane", "questionnaire", "htmlactivity", "oucollaborate",
        "dataplus", "glossary", "repeatactivity", "resource", "forumng", "url",
        "subpage", "oucontent", "homepage"
    ]
    return columnas_actividad

# Función para preprocesar el dataset
def preprocess_data(df_tmp):
    """
    Preprocesa el dataset eliminando duplicados, conservando columnas esenciales,
    y validando la integridad de los datos.
    
    Raises:
        ValueError: Si faltan columnas críticas o el dataframe está vacío
        TypeError: Si los tipos de datos no son válidos
    """
    print("Inicio del preprocesamiento de datos...")
    
    # VALIDACIÓN 1: DataFrames vacío
    if df_tmp is None or len(df_tmp) == 0:
        raise ValueError("[ERROR] El dataset está vacío. No hay filas para procesar.")
    
    # VALIDACIÓN 2: Columnas críticas
    columnas_criticas = ['id_student', 'date', 'final_result']
    columnas_faltantes = [col for col in columnas_criticas if col not in df_tmp.columns]
    if columnas_faltantes:
        raise ValueError(f"[ERROR] Faltan columnas críticas: {', '.join(columnas_faltantes)}")
    
    print(f"[OK] Validaciones iniciales pasadas")
    print(f"  - Filas del archivo: {len(df_tmp)}")
    print(f"  - Columnas encontradas: {len(df_tmp.columns)}")

    df_tmp.columns = df_tmp.columns.str.strip() 
    
    # Convertir la columna "final_result" en la columna binaria: 1 = abandono, 0 = no abandono
    try:
        df_tmp['final_result'] = df_tmp['final_result'].astype(str).str.strip().str.lower()
        df_tmp['final_result'] = df_tmp['final_result'].apply(lambda x: 1 if x == 'withdrawn' else 0)
    except Exception as e:
        raise TypeError(f"[ERROR] No se pudo procesar la columna 'final_result': {str(e)}")

    # Ordenar por id_student y date en orden descendente (de más reciente a más antigua)
    df_tmp = df_tmp.sort_values(by=['id_student', 'date'], ascending=[True, False])

    # Eliminar la columna 'Unnamed: 0' si no es necesaria
    if 'Unnamed: 0' in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=['Unnamed: 0'])

    # Obtener columnas de actividad antes de definir la función
    # IMPORTANTE: Filtrar solo las columnas que existen en el dataframe
    todas_columnas_actividad = clean_columns(df_tmp)
    columnas_actividad_validas = [col for col in todas_columnas_actividad if col in df_tmp.columns]

    # Función para seleccionar la fila adecuada por estudiante
    def seleccionar_fila(grupo):
        if grupo['final_result'].iloc[0] == 0:
            fila_269 = grupo[grupo['date'] == 269]
            if not fila_269.empty:
                return fila_269.iloc[0]
            else:
                return grupo.iloc[0]  # Si no hay date=269, se toma la más reciente
        else:
            # Si hay columnas de actividad, buscar fila con actividad
            if len(columnas_actividad_validas) > 0:
                grupo_con_actividad = grupo[(grupo[columnas_actividad_validas] > 0.0).any(axis=1)]
                if not grupo_con_actividad.empty:
                    return grupo_con_actividad.iloc[0]
            # Si no tiene actividad o no hay columnas de actividad, tomar fila con date=0
            fila_0 = grupo[grupo['date'] == 0]
            if not fila_0.empty:
                return fila_0.iloc[0]
            else:
                return grupo.iloc[-1]

    # Eliminar duplicados quedándonos con la fila más reciente por cada id_student
    df_final = df_tmp.groupby('id_student', group_keys=False).apply(seleccionar_fila)

    # Convertir columnas a int (excepto las de actividad que pueden ser flotantes)
    # Usar 'Int64' (con I mayúscula) para permitir valores NaN
    for col in df_final.columns:
        if col not in columnas_actividad_validas:
            try:
                # Intentar convertir a Int64 (nullable integer)
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce').astype('Int64')
            except Exception as e:
                print(f"Advertencia: No se pudo convertir {col} a int: {e}")
                # Dejar la columna como está si hay problemas

    # Columnas a conservar: id_student, date, final_result, atributos binarios y nuevos atributos no-binarios
    columnas_a_conservar = [
        'id_student', 'date', 'final_result', 
        'gender_F', 'disability_N',
        # Nuevos atributos no-binarios
        'age_band_0-35', 'age_band_35-55', 'age_band_55<=',
        'highest_education_A Level or Equivalent', 'highest_education_HE Qualification',
        'highest_education_Lower Than A Level', 'highest_education_No Formal quals',
        'highest_education_Post Graduate Qualification',
        'studied_credits'
    ]

    # VALIDACIÓN 3: Verificar que existen todas las columnas esperadas
    columnas_faltantes = [col for col in columnas_a_conservar if col not in df_final.columns]
    if columnas_faltantes:
        print(f"[ADVERTENCIA] Faltan columnas esperadas: {', '.join(columnas_faltantes)}")
        print(f"   Se conservarán solo las columnas disponibles")
        columnas_a_conservar = [col for col in columnas_a_conservar if col in df_final.columns]
    
    # Verificamos que existen antes de borrarlas
    columnas_a_eliminar = [col for col in df_final.columns if col not in columnas_a_conservar]

    # Las borramos
    df_final = df_final.drop(columns=columnas_a_eliminar)

    # ✅ ERROR #7: Validación de valores NaN en columnas críticas
    critical_cols = ['id_student', 'date', 'final_result']
    nan_en_criticas = []
    for col in critical_cols:
        if col in df_final.columns:
            nan_count = df_final[col].isna().sum()
            if nan_count > 0:
                nan_en_criticas.append((col, nan_count))
    
    if nan_en_criticas:
        print(f"[ADVERTENCIA] Valores NaN en columnas críticas:")
        for col, count in nan_en_criticas:
            print(f"   - {col}: {count} valores NaN")
        # Remover filas con NaN en columnas críticas
        df_final = df_final.dropna(subset=critical_cols)
        print(f"   → Filas con NaN removidas. Filas restantes: {df_final.shape[0]}")
    
    # [OK] ERROR #7: Reporte de NaN en todas las columnas
    nan_summary = df_final.isna().sum()
    if nan_summary.sum() > 0:
        print(f"\n[ADVERTENCIA] Valores NaN por columna:")
        for col, count in nan_summary[nan_summary > 0].items():
            print(f"   - {col}: {count} valores NaN ({count/len(df_final)*100:.1f}%)")

    # LOG: Mostrar resumen de la limpieza realizada
    print(f"\n[OK] Preprocesamiento completado:")
    print(f"  - Filas procesadas: {df_final.shape[0]}")
    print(f"  - Columnas conservadas: {df_final.shape[1]}")
    print(f"  - Columnas eliminadas: {len(columnas_a_eliminar)}")
    print(f"  - Distribución final_result: {df_final['final_result'].value_counts().to_dict()}")
    print(f"  - Columnas de highest_education presentes: {[c for c in df_final.columns if 'highest_education' in c]}")
    print(f"  - Todas las columnas conservadas: {list(df_final.columns)}\n")

    return df_final
