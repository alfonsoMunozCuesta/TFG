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
    print("Inicio del preprocesamiento de datos...")

    df_tmp.columns = df_tmp.columns.str.strip() 
    # Convertir la columna "final_result" en la columna binaria: 1 = abandono, 0 = no abandono
    df_tmp['final_result'] = df_tmp['final_result'].str.strip().str.lower()  # Eliminando espacios y mayúsculas
    df_tmp['final_result'] = df_tmp['final_result'].apply(lambda x: 1 if x == 'withdrawn' else 0)

    # Ordenar por id_student y date en orden descendente (de más reciente a más antigua)
    df_tmp = df_tmp.sort_values(by=['id_student', 'date'], ascending=[True, False])

    # Eliminar la columna 'Unnamed: 0' si no es necesaria
    if 'Unnamed: 0' in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=['Unnamed: 0'])

    # Función para seleccionar la fila adecuada por estudiante
    def seleccionar_fila(grupo):
        #columnas_actividad_validas = clean_columns(df_tmp)  # Aquí se usa df_tmp, no df
        if grupo['final_result'].iloc[0] == 0:
            fila_269 = grupo[grupo['date'] == 269]
            if not fila_269.empty:
                return fila_269.iloc[0]
            else:
                return grupo.iloc[0]  # Si no hay date=269, se toma la más reciente
        else:
            grupo_con_actividad = grupo[(grupo[columnas_actividad_validas] > 0.0).any(axis=1)]
            if not grupo_con_actividad.empty:
                return grupo_con_actividad.iloc[0]
            else:  # No tiene actividad, por tanto devolvemos la fila con date=0
                fila_0 = grupo[grupo['date'] == 0]
                if not fila_0.empty:
                    return fila_0.iloc[0]  # Por seguridad, aunque no debería pasar
                else:
                    return grupo.iloc[-1]

    # Eliminar duplicados quedándonos con la fila más reciente por cada id_student
    df_final = df_tmp.groupby('id_student', group_keys=False).apply(seleccionar_fila)

    # Convertir todas las columnas a enteros, excepto las de actividad 
    columnas_actividad_validas = clean_columns(df_tmp)

    for col in df_final.columns:
        if col not in columnas_actividad_validas:
            df_final[col] = df_final[col].astype(int)

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

    # Verificamos que existen antes de borrarlas
    columnas_a_eliminar = [col for col in df_final.columns if col not in columnas_a_conservar]

    # Las borramos
    df_final = df_final.drop(columns=columnas_a_eliminar)

    return df_final
