from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_DATA_PATH = DATA_DIR / "temp_data.csv"
CLEAN_DATA_PATH = BASE_DIR / "dataset_limpio.csv"

LLAMA_SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

df = None
df_limpio = None


def load_temp_data():
    global df
    if df is None:
        df = pd.read_csv(TEMP_DATA_PATH, sep=";")
    return df


def load_clean_data():
    global df_limpio
    if df_limpio is None:
        df_limpio = pd.read_csv(CLEAN_DATA_PATH, sep=";")
    return df_limpio
