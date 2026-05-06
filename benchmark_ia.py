"""
Benchmark sencillo del modulo de IA local.

Mide tiempos de respuesta contra el endpoint configurado en config.py y, en
Windows, intenta estimar consumo de RAM/CPU del proceso llama-server.
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

import requests

from config import LLAMA_SERVER_URL, MODEL_NAME


DEFAULT_PROMPT = (
    "Interpreta de forma breve y academica estos resultados de supervivencia: "
    "1200 estudiantes, 310 eventos de abandono, follow-up medio de 145 dias, "
    "Hazard Ratio de 1.42 para creditos estudiados y p-valor de 0.03."
)

TECHNIQUE_PROMPTS = [
    {
        "technique": "Kaplan-Meier",
        "prompt": (
            "Interpreta de forma breve y academica un analisis Kaplan-Meier con "
            "1200 estudiantes, 310 abandonos observados, supervivencia estimada "
            "a los 100 dias de 0.82 y a los 200 dias de 0.69. La curva del grupo "
            "con mas creditos estudiados desciende antes que la del resto."
        ),
    },
    {
        "technique": "Regresion de Cox",
        "prompt": (
            "Interpreta de forma breve y academica una regresion de Cox aplicada "
            "al abandono academico. La variable creditos estudiados presenta un "
            "Hazard Ratio de 1.42 con p-valor 0.03, edad 35-55 tiene Hazard Ratio "
            "0.91 con p-valor 0.28 y discapacidad presenta Hazard Ratio 1.18 con "
            "p-valor 0.12."
        ),
    },
    {
        "technique": "Test Log-Rank",
        "prompt": (
            "Interpreta de forma breve y academica un Test Log-Rank que compara "
            "curvas de supervivencia por nivel educativo previo. El estadistico "
            "de prueba es 12.84 y el p-valor es 0.004. Las curvas muestran una "
            "separacion visible entre grupos a partir de la mitad del seguimiento."
        ),
    },
    {
        "technique": "Weibull",
        "prompt": (
            "Interpreta de forma breve y academica un modelo Weibull para abandono "
            "academico. El parametro de forma rho es 1.31, la mediana de supervivencia "
            "estimada es 176 dias, el log-likelihood es -842.5 y el AIC es 1689.0. "
            "El riesgo parece aumentar ligeramente con el tiempo."
        ),
    },
    {
        "technique": "Exponencial",
        "prompt": (
            "Interpreta de forma breve y academica un modelo Exponencial para "
            "abandono academico. El parametro lambda estimado es 0.0048, la mediana "
            "de supervivencia es 144 dias, el log-likelihood es -861.2 y el AIC es "
            "1724.4. El modelo asume riesgo constante durante el seguimiento."
        ),
    },
    {
        "technique": "Random Survival Forest",
        "prompt": (
            "Interpreta de forma breve y academica un Random Survival Forest aplicado "
            "al abandono academico. El indice de concordancia es 0.71, la puntuacion "
            "out-of-bag es 0.66 y las variables mas importantes son creditos estudiados, "
            "edad y nivel educativo previo. Los perfiles de alto riesgo muestran una "
            "curva de supervivencia claramente inferior."
        ),
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mide tiempos y consumo aproximado del servidor local de IA."
    )
    parser.add_argument(
        "--mode",
        choices=("techniques", "repeat"),
        default="techniques",
        help=(
            "techniques mide una consulta por tecnica; repeat repite el mismo "
            "prompt varias veces."
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
    parser.add_argument("--endpoint", default=LLAMA_SERVER_URL, help="Endpoint compatible con chat completions.")
    parser.add_argument("--model", default=MODEL_NAME, help="Nombre del modelo que recibira el endpoint.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt usado en cada consulta.")
    parser.add_argument(
        "--process-name",
        default="llama-server",
        help="Nombre del proceso que se usara para estimar RAM/CPU.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmarks",
        help="Carpeta donde se guardaran CSV y JSON con los resultados.",
    )
    return parser.parse_args()


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


def _build_payload(prompt: str, model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente academico que interpreta resultados de "
                    "analisis de supervivencia de forma clara, prudente y breve."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 320,
        "stream": False,
    }


def _call_model(prompt: str, timeout: int, endpoint: str, model: str) -> tuple[str, float]:
    start = time.perf_counter()
    response = requests.post(endpoint, json=_build_payload(prompt, model), timeout=timeout)
    elapsed = time.perf_counter() - start
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return content, elapsed


def _run_once(
    index: int,
    prompt: str,
    timeout: int,
    process_name: str,
    endpoint: str,
    model: str,
    technique: str,
) -> dict:
    before = _query_process_snapshot(process_name)
    content, elapsed = _call_model(prompt, timeout, endpoint, model)
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


def _save_results(results: list[dict], args: argparse.Namespace, summary: dict) -> tuple[Path, Path]:
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
        "mode": args.mode,
        "runs": args.runs,
        "warmup": args.warmup,
        "prompt": args.prompt if args.mode == "repeat" else None,
        "technique_prompts": TECHNIQUE_PROMPTS if args.mode == "techniques" else None,
        "summary": summary,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def _build_summary(results: list[dict]) -> dict:
    times = [row["elapsed_seconds"] for row in results]
    cpu_values = [row["cpu_percent_approx"] for row in results if row["cpu_percent_approx"] is not None]
    ram_values = [row["ram_mb_approx"] for row in results if row["ram_mb_approx"] is not None]

    summary = {
        "time_mean_seconds": round(statistics.mean(times), 3),
        "time_median_seconds": round(statistics.median(times), 3),
        "time_max_seconds": round(max(times), 3),
        "time_min_seconds": round(min(times), 3),
        "response_chars_mean": round(statistics.mean(row["response_chars"] for row in results), 1),
        "ram_mb_max_approx": round(max(ram_values), 2) if ram_values else None,
        "cpu_percent_mean_approx": round(statistics.mean(cpu_values), 2) if cpu_values else None,
        "cpu_percent_max_approx": round(max(cpu_values), 2) if cpu_values else None,
    }
    return summary


def _build_tasks(args: argparse.Namespace) -> list[dict]:
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
    for item in TECHNIQUE_PROMPTS:
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
    args = _parse_args()

    if args.runs <= 0:
        print("ERROR: --runs debe ser mayor que 0.")
        return 1

    print("Benchmark IA local")
    print(f"Endpoint: {args.endpoint}")
    print(f"Modelo:   {args.model}")
    print(f"Modo:     {args.mode}")
    if args.mode == "techniques":
        print(f"Consultas: {len(TECHNIQUE_PROMPTS)} tecnicas x {args.runs} repeticion(es)")
    else:
        print(f"Runs:     {args.runs} (+ {args.warmup} warmup)")
    print()

    try:
        for index in range(args.warmup):
            print(f"Warmup {index + 1}/{args.warmup}...")
            _call_model(args.prompt, args.timeout, args.endpoint, args.model)
    except Exception as exc:
        print("ERROR: no se pudo contactar con el servidor de IA.")
        print(f"Detalle: {exc}")
        print("Comprueba que START_LLAMA_SERVER.bat esta ejecutandose.")
        return 1

    tasks = _build_tasks(args)
    results = []
    for position, task in enumerate(tasks, start=1):
        print(f"{task['technique']} ({position}/{len(tasks)})...", end=" ", flush=True)
        try:
            row = _run_once(
                task["run"],
                task["prompt"],
                args.timeout,
                args.process_name,
                args.endpoint,
                args.model,
                task["technique"],
            )
        except Exception as exc:
            print("ERROR")
            print(f"Detalle: {exc}")
            return 1
        results.append(row)
        print(f"{row['elapsed_seconds']} s")

    summary = _build_summary(results)
    csv_path, json_path = _save_results(results, args, summary)

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
