"""Configuracion comun del proyecto.

Centraliza rutas, parametros del servidor local de IA y carga perezosa de los
datasets para que otros modulos no repitan constantes ni lecturas de disco.
"""

from pathlib import Path

import pandas as pd


# Todas las rutas se calculan desde la carpeta del proyecto para que la app
# funcione aunque se lance desde un directorio de trabajo distinto.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_DATA_PATH = DATA_DIR / "temp_data.csv"
CLEAN_DATA_PATH = BASE_DIR / "dataset_limpio.csv"

# Endpoint compatible con la API de OpenAI expuesto por el servidor local.
LLAMA_SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

# Caches en memoria para no recargar CSV en cada llamada.
df = None
df_limpio = None


def load_temp_data():
    """Carga el dataset temporal original desde disco."""
    global df
    if df is None:
        df = pd.read_csv(TEMP_DATA_PATH, sep=";")
    return df


def load_clean_data():
    """Carga el dataset limpio que alimenta los analisis."""
    global df_limpio
    if df_limpio is None:
        df_limpio = pd.read_csv(CLEAN_DATA_PATH, sep=";")
    return df_limpio
