"""
config.py - Módulo de configuración centralizada.
Carga única de datos y constantes compartidas por toda la aplicación.
"""
from pathlib import Path
import pandas as pd

# ── Rutas ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ── Constantes del LLM (Qwen2.5 vía llama-server) ─────────────────────
LLAMA_SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

# ── Carga única de datasets ───────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "temp_data.csv", sep=";")
df_limpio = pd.read_csv(BASE_DIR / "dataset_limpio.csv", sep=";")
