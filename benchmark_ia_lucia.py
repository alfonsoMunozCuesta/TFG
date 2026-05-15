"""
Benchmark sencillo de la IA local para las tecnicas existentes en este Dash.

Mide las mismas metricas de rendimiento que el benchmark original:
tiempo de respuesta, longitud de respuesta, RAM aproximada y CPU aproximada.
Los prompts se construyen solo con las tecnicas y variables disponibles en
esta aplicacion: Kaplan-Meier, Regresion de Cox y Test Log-Rank.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter, KaplanMeierFitter

from log_rank_test import perform_log_rank_test


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "dataset_limpio.csv"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "llama3:latest"

DASH_VARIABLES = [
    {
        "column": "gender_F",
        "label": "Genero",
        "groups": {0: "Masculino", 1: "Femenino"},
    },
    {
        "column": "disability_N",
        "label": "Discapacidad",
        "groups": {0: "Sin discapacidad", 1: "Con discapacidad"},
    },
]

DEFAULT_PROMPT = (
    "Interpreta de forma breve y academica los resultados de abandono del "
    "dashboard de supervivencia usando Kaplan-Meier, Regresion de Cox o "
    "Test Log-Rank cuando corresponda."
)


def _parse_args() -> argparse.Namespace:
    """Define y lee los argumentos de consola para configurar el benchmark."""
    parser = argparse.ArgumentParser(
        description="Mide tiempos y consumo aproximado del servidor local de IA."
    )
    parser.add_argument(
        "--mode",
        choices=("techniques", "repeat"),
        default="techniques",
        help=(
            "techniques mide consultas para las tecnicas del Dash; repeat repite "
            "el mismo prompt varias veces."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Repeticiones por tecnica en modo techniques, o numero de consultas en modo repeat.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Consultas previas no medidas.")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout por consulta en segundos.")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Endpoint compatible con chat completions.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nombre del modelo.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt usado en modo repeat.")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=512,
        help="Tamano de contexto para Ollama. Menor valor reduce memoria de KV cache.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=180,
        help="Maximo de tokens generados por respuesta.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="CSV limpio usado por el Dash.",
    )
    parser.add_argument(
        "--process-name",
        default="ollama",
        help="Nombre del proceso que se usara para estimar RAM/CPU.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmarks",
        help="Carpeta donde se guardaran CSV y JSON con los resultados.",
    )
    return parser.parse_args()


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    """Carga el dataset limpio con las columnas minimas que usa el Dash."""
    df = pd.read_csv(dataset_path, sep=";")
    required = {"date", "final_result", "gender_F", "disability_N"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {', '.join(missing)}")
    return df


def _query_process_snapshot(process_name: str) -> dict | None:
    """Devuelve CPU acumulada y RAM del proceso en Windows, si se encuentra."""
    if platform.system().lower() != "windows":
        return None

    ps_script = f"""
    $p = Get-Process -ErrorAction SilentlyContinue |
        Where-Object {{ $_.ProcessName -like '*{process_name}*' }} |
        Sort-Object WorkingSet64 -Descending |
        Select-Object -First 1 Id,ProcessName,CPU,WorkingSet64
    if ($null -ne $p) {{ $p | ConvertTo-Json -Compress }}
    """

    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None

    output = completed.stdout.strip()
    if not output:
        return None

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None

    return {
        "pid": data.get("Id"),
        "name": data.get("ProcessName"),
        "cpu_seconds": float(data.get("CPU") or 0.0),
        "ram_mb": round(float(data.get("WorkingSet64") or 0.0) / (1024 * 1024), 2),
    }


def _build_payload(prompt: str, model: str, endpoint: str, num_ctx: int, max_tokens: int) -> dict:
    """Construye el cuerpo JSON que se envia al endpoint del modelo."""
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente academico que interpreta resultados de "
                "analisis de supervivencia de forma clara, prudente y breve."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if endpoint.rstrip("/").endswith("/api/chat"):
        return {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }


def _call_model(
    prompt: str,
    timeout: int,
    endpoint: str,
    model: str,
    num_ctx: int,
    max_tokens: int,
) -> tuple[str, float]:
    """Llama al modelo, mide el tiempo de respuesta y devuelve el texto generado."""
    start = time.perf_counter()
    response = requests.post(
        endpoint,
        json=_build_payload(prompt, model, endpoint, num_ctx, max_tokens),
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{exc}. Respuesta del servidor: {response.text}") from exc

    data = response.json()
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
    else:
        content = data["message"]["content"]
    return content, elapsed


def _survival_probability(df: pd.DataFrame, day: int) -> float:
    """Calcula la supervivencia Kaplan-Meier aproximada en un dia concreto."""
    kmf = KaplanMeierFitter()
    kmf.fit(df["date"], event_observed=df["final_result"])
    return float(kmf.predict(day))


def _format_counts(df: pd.DataFrame, column: str, groups: dict[int, str]) -> str:
    """Devuelve un resumen legible de frecuencias por grupo."""
    counts = df[column].value_counts().sort_index()
    return ", ".join(f"{groups.get(int(value), value)}: {int(count)}" for value, count in counts.items())


def _build_kaplan_prompt(df: pd.DataFrame, variable: dict | None = None) -> dict:
    """Crea un prompt de Kaplan-Meier global o segmentado por covariable."""
    total = len(df)
    events = int(df["final_result"].sum())
    min_day = int(df["date"].min())
    max_day = int(df["date"].max())
    mean_day = round(float(df["date"].mean()), 2)
    surv_100 = round(_survival_probability(df, 100), 3)
    surv_200 = round(_survival_probability(df, 200), 3)

    if variable is None:
        group_parts = []
        for item in DASH_VARIABLES:
            group_parts.append(
                f"{item['label']} ({_format_counts(df, item['column'], item['groups'])})"
            )
        return {
            "technique": "Kaplan-Meier",
            "prompt": (
                "Interpreta de forma breve y academica la curva Kaplan-Meier global "
                "del dashboard de abandono academico. "
                f"Dataset: {total} estudiantes, {events} eventos de abandono, "
                f"seguimiento entre {min_day} y {max_day} dias, media {mean_day} dias. "
                f"La supervivencia estimada es {surv_100} en el dia 100 y {surv_200} "
                "en el dia 200. Las variables disponibles para segmentar las curvas "
                f"en este Dash son: {'; '.join(group_parts)}."
            ),
        }

    column = variable["column"]
    groups = variable["groups"]
    group_parts = []
    for value, label in groups.items():
        subset = df[df[column] == value]
        if subset.empty:
            continue
        group_parts.append(
            f"{label}: n={len(subset)}, abandonos={int(subset['final_result'].sum())}, "
            f"S(100)={round(_survival_probability(subset, 100), 3)}, "
            f"S(200)={round(_survival_probability(subset, 200), 3)}"
        )

    return {
        "technique": "Kaplan-Meier",
        "prompt": (
            "Interpreta de forma breve y academica las curvas Kaplan-Meier del "
            f"dashboard comparando por {variable['label']}. "
            f"Resumen por grupo: {'; '.join(group_parts)}."
        ),
    }


def _build_cox_prompt(df: pd.DataFrame, covariables: list[str], variable_label: str) -> dict:
    """Crea un prompt de Regresion de Cox con covariables reales del Dash."""
    df_cox = df[["date", "final_result"] + covariables].copy()
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="date", event_col="final_result")
    summary = cph.summary.reset_index()

    parts = []
    for _, row in summary.iterrows():
        parts.append(
            f"{row['covariate']}: HR={round(float(row['exp(coef)']), 3)}, "
            f"coef={round(float(row['coef']), 3)}, p={round(float(row['p']), 4)}"
        )

    return {
        "technique": "Regresion de Cox",
        "prompt": (
            "Interpreta de forma breve y academica una Regresion de Cox del "
            "dashboard de abandono academico. "
            f"Covariables incluidas: {', '.join(covariables)}. "
            f"Resultados: {'; '.join(parts)}. "
            "Explica como afectan las covariables a la probabilidad de abandono."
        ),
    }


def _build_logrank_prompt(df: pd.DataFrame) -> dict:
    """Crea un prompt de Test Log-Rank con las covariables reales del Dash."""
    parts = []
    for variable in DASH_VARIABLES:
        result = perform_log_rank_test(df, variable["column"])
        row = result.iloc[0]
        parts.append(
            f"{variable['label']} ({_format_counts(df, variable['column'], variable['groups'])}): "
            f"estadistico={round(float(row['test_statistic']), 4)}, "
            f"p-valor={round(float(row['p_value']), 6)}, "
            f"decision={row['Decisión']}, conclusion={row['Conclusión']}"
        )

    return {
        "technique": "Test Log-Rank",
        "prompt": (
            "Interpreta de forma breve y academica un Test Log-Rank del dashboard "
            "comparando curvas de supervivencia con las variables existentes. "
            f"Resultados: {'; '.join(parts)}."
        ),
    }


def _build_dash_prompts(df: pd.DataFrame) -> list[dict]:
    """Prepara prompts solo para las tecnicas y variables existentes en el Dash."""
    prompts = [_build_kaplan_prompt(df)]

    prompts.append(
        _build_cox_prompt(
            df,
            [variable["column"] for variable in DASH_VARIABLES],
            "Genero + Discapacidad",
        )
    )

    prompts.append(_build_logrank_prompt(df))

    return prompts


def _run_once(
    index: int,
    prompt: str,
    timeout: int,
    process_name: str,
    endpoint: str,
    model: str,
    num_ctx: int,
    max_tokens: int,
    technique: str,
) -> dict:
    """Ejecuta una prueba individual y registra metricas del sistema."""
    before = _query_process_snapshot(process_name)
    content, elapsed = _call_model(prompt, timeout, endpoint, model, num_ctx, max_tokens)
    after = _query_process_snapshot(process_name)

    cpu_percent = None
    ram_mb = None
    if before and after and before.get("pid") == after.get("pid") and elapsed > 0:
        cpu_delta = max(0.0, after["cpu_seconds"] - before["cpu_seconds"])
        cpu_percent = round((cpu_delta / elapsed / max(1, os.cpu_count() or 1)) * 100, 2)
        ram_mb = after["ram_mb"]
    elif after:
        ram_mb = after["ram_mb"]

    return {
        "run": index,
        "technique": technique,
        "elapsed_seconds": round(elapsed, 3),
        "response_chars": len(content),
        "cpu_percent_approx": cpu_percent,
        "ram_mb_approx": ram_mb,
        "process": after["name"] if after else None,
        "pid": after["pid"] if after else None,
    }


def _save_results(
    results: list[dict],
    args: argparse.Namespace,
    summary: dict,
    technique_prompts: list[dict],
) -> tuple[Path, Path]:
    """Guarda los resultados detallados y el resumen en disco."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ia_benchmark_{timestamp}.csv"
    json_path = out_dir / f"ia_benchmark_{timestamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "model": args.model,
        "num_ctx": args.num_ctx,
        "max_tokens": args.max_tokens,
        "mode": args.mode,
        "runs": args.runs,
        "warmup": args.warmup,
        "dataset": args.dataset,
        "prompt": args.prompt if args.mode == "repeat" else None,
        "technique_prompts": technique_prompts if args.mode == "techniques" else None,
        "summary": summary,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def _build_summary(results: list[dict]) -> dict:
    """Resume las metricas principales obtenidas en todas las ejecuciones."""
    times = [row["elapsed_seconds"] for row in results]
    cpu_values = [row["cpu_percent_approx"] for row in results if row["cpu_percent_approx"] is not None]
    ram_values = [row["ram_mb_approx"] for row in results if row["ram_mb_approx"] is not None]

    return {
        "time_mean_seconds": round(statistics.mean(times), 3),
        "time_median_seconds": round(statistics.median(times), 3),
        "time_max_seconds": round(max(times), 3),
        "time_min_seconds": round(min(times), 3),
        "response_chars_mean": round(statistics.mean(row["response_chars"] for row in results), 1),
        "ram_mb_max_approx": round(max(ram_values), 2) if ram_values else None,
        "cpu_percent_mean_approx": round(statistics.mean(cpu_values), 2) if cpu_values else None,
        "cpu_percent_max_approx": round(max(cpu_values), 2) if cpu_values else None,
    }


def _build_tasks(args: argparse.Namespace, technique_prompts: list[dict]) -> list[dict]:
    """Prepara la lista de tareas que se lanzaran durante el benchmark."""
    if args.mode == "repeat":
        return [
            {
                "run": index,
                "technique": "Prompt repetido",
                "prompt": args.prompt,
            }
            for index in range(1, args.runs + 1)
        ]

    tasks = []
    run_index = 1
    for item in technique_prompts:
        for _ in range(args.runs):
            tasks.append(
                {
                    "run": run_index,
                    "technique": item["technique"],
                    "prompt": item["prompt"],
                }
            )
            run_index += 1
    return tasks


def main() -> int:
    """Orquesta el benchmark completo desde argumentos hasta ficheros finales."""
    args = _parse_args()

    if args.runs <= 0:
        print("ERROR: --runs debe ser mayor que 0.")
        return 1

    try:
        df = _load_dataset(args.dataset)
        technique_prompts = _build_dash_prompts(df)
    except Exception as exc:
        print("ERROR: no se pudieron preparar las tecnicas del Dash.")
        print(f"Detalle: {exc}")
        return 1

    print("Benchmark IA local del Dash")
    print(f"Endpoint: {args.endpoint}")
    print(f"Modelo:   {args.model}")
    print(f"Contexto: {args.num_ctx}")
    print(f"Tokens:   {args.max_tokens}")
    print(f"Dataset:  {args.dataset}")
    print(f"Modo:     {args.mode}")
    if args.mode == "techniques":
        print(f"Consultas: {len(technique_prompts)} tecnicas x {args.runs} repeticion(es)")
        print("Tecnicas:")
        for item in technique_prompts:
            print(f"- {item['technique']}")
    else:
        print(f"Runs:     {args.runs} (+ {args.warmup} warmup)")
    print()

    try:
        for index in range(args.warmup):
            print(f"Warmup {index + 1}/{args.warmup}...")
            _call_model(
                args.prompt,
                args.timeout,
                args.endpoint,
                args.model,
                args.num_ctx,
                args.max_tokens,
            )
    except Exception as exc:
        print("ERROR: no se pudo contactar con el servidor de IA.")
        print(f"Detalle: {exc}")
        print("Comprueba que Ollama esta ejecutandose y que el modelo esta descargado.")
        return 1

    tasks = _build_tasks(args, technique_prompts)
    results = []
    for position, task in enumerate(tasks, start=1):
        print(
            f"{task['technique']} ({position}/{len(tasks)})...",
            end=" ",
            flush=True,
        )
        try:
            row = _run_once(
                task["run"],
                task["prompt"],
                args.timeout,
                args.process_name,
                args.endpoint,
                args.model,
                args.num_ctx,
                args.max_tokens,
                task["technique"],
            )
        except Exception as exc:
            print("ERROR")
            print(f"Detalle: {exc}")
            return 1
        results.append(row)
        print(f"{row['elapsed_seconds']} s")

    summary = _build_summary(results)
    csv_path, json_path = _save_results(results, args, summary, technique_prompts)

    print()
    print("Resumen")
    print(f"- Tiempo medio:  {summary['time_mean_seconds']} s")
    print(f"- Tiempo maximo: {summary['time_max_seconds']} s")
    print(f"- RAM aprox.:    {summary['ram_mb_max_approx'] or 'N/A'} MB")
    print(f"- CPU aprox.:    {summary['cpu_percent_mean_approx'] or 'N/A'} % medio")
    print()
    print("Detalle por tecnica")
    for row in results:
        print(
            f"- {row['technique']}: {row['elapsed_seconds']} s, "
            f"RAM {row['ram_mb_approx'] or 'N/A'} MB, "
            f"CPU {row['cpu_percent_approx'] or 'N/A'} %"
        )
    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
