"""
Benchmark reproducible para llama-server (OpenAI-compatible endpoint).
Mide latencia, throughput aproximado y uso de CPU/RAM bajo carga.

Uso:
  python benchmark_llama_server.py
  python benchmark_llama_server.py --requests 12 --concurrency 3 --max-tokens 180
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import psutil
import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-1.5b-instruct"


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    approx_tokens_out: int
    error: str = ""


def approx_token_count(text: str) -> int:
    # Aproximacion simple y estable para comparar runs entre si.
    return max(1, int(len(text) / 4))


def call_once(session: requests.Session, prompt: str, max_tokens: int, temperature: float, timeout_s: int) -> RequestResult:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    t0 = time.perf_counter()
    try:
        response = session.post(URL, json=payload, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        latency = time.perf_counter() - t0
        return RequestResult(ok=True, latency_s=latency, approx_tokens_out=approx_token_count(content))
    except Exception as exc:
        latency = time.perf_counter() - t0
        return RequestResult(ok=False, latency_s=latency, approx_tokens_out=0, error=str(exc))


def find_llama_process() -> Optional[psutil.Process]:
    for proc in psutil.process_iter(attrs=["name", "exe", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "llama-server" in name or "llama-server" in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def monitor_resources(stop_event: threading.Event, sample_every_s: float = 0.5) -> dict:
    llama_proc = find_llama_process()
    cpu_samples = []
    ram_samples_mb = []
    proc_cpu_samples = []
    proc_ram_samples_mb = []

    if llama_proc is not None:
        try:
            llama_proc.cpu_percent(interval=None)
        except Exception:
            llama_proc = None

    while not stop_event.is_set():
        cpu_samples.append(psutil.cpu_percent(interval=None))
        ram_samples_mb.append(psutil.virtual_memory().used / (1024 * 1024))

        if llama_proc is not None:
            try:
                proc_cpu_samples.append(llama_proc.cpu_percent(interval=None))
                proc_ram_samples_mb.append(llama_proc.memory_info().rss / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        time.sleep(sample_every_s)

    return {
        "cpu_samples": cpu_samples,
        "ram_samples_mb": ram_samples_mb,
        "proc_cpu_samples": proc_cpu_samples,
        "proc_ram_samples_mb": proc_ram_samples_mb,
        "llama_detected": llama_proc is not None,
    }


def summarize(results: List[RequestResult], elapsed_wall_s: float, res: dict) -> dict:
    oks = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]

    latencies = [r.latency_s for r in oks]
    total_tokens = sum(r.approx_tokens_out for r in oks)
    throughput_rps = len(oks) / elapsed_wall_s if elapsed_wall_s > 0 else 0.0
    throughput_tps = total_tokens / elapsed_wall_s if elapsed_wall_s > 0 else 0.0

    def p95(xs: List[float]) -> float:
        if not xs:
            return 0.0
        if len(xs) == 1:
            return xs[0]
        xs_sorted = sorted(xs)
        idx = int(0.95 * (len(xs_sorted) - 1))
        return xs_sorted[idx]

    cpu_samples = res["cpu_samples"]
    ram_samples_mb = res["ram_samples_mb"]
    proc_cpu_samples = res["proc_cpu_samples"]
    proc_ram_samples_mb = res["proc_ram_samples_mb"]

    return {
        "ok": len(oks),
        "errors": len(errs),
        "error_examples": [e.error for e in errs[:3]],
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p95_s": p95(latencies),
        "latency_max_s": max(latencies) if latencies else 0.0,
        "approx_tokens_total": total_tokens,
        "throughput_req_s": throughput_rps,
        "throughput_tok_s": throughput_tps,
        "system_cpu_mean_pct": statistics.mean(cpu_samples) if cpu_samples else 0.0,
        "system_cpu_max_pct": max(cpu_samples) if cpu_samples else 0.0,
        "system_ram_mean_mb": statistics.mean(ram_samples_mb) if ram_samples_mb else 0.0,
        "system_ram_max_mb": max(ram_samples_mb) if ram_samples_mb else 0.0,
        "llama_detected": res["llama_detected"],
        "llama_cpu_mean_pct": statistics.mean(proc_cpu_samples) if proc_cpu_samples else 0.0,
        "llama_cpu_max_pct": max(proc_cpu_samples) if proc_cpu_samples else 0.0,
        "llama_ram_mean_mb": statistics.mean(proc_ram_samples_mb) if proc_ram_samples_mb else 0.0,
        "llama_ram_max_mb": max(proc_ram_samples_mb) if proc_ram_samples_mb else 0.0,
        "elapsed_wall_s": elapsed_wall_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reproducible para llama-server")
    parser.add_argument("--requests", type=int, default=10, help="Numero total de requests")
    parser.add_argument("--concurrency", type=int, default=2, help="Requests concurrentes")
    parser.add_argument("--max-tokens", type=int, default=180, help="max_tokens por request")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperatura")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout por request")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Ruta JSON de salida")
    args = parser.parse_args()

    prompt = (
        "Resume en 5 frases las conclusiones clave de un analisis de supervivencia, "
        "sin inventar datos y con tono tecnico claro."
    )

    # Calentamiento
    with requests.Session() as s:
        warm = call_once(s, prompt, max_tokens=min(64, args.max_tokens), temperature=args.temperature, timeout_s=args.timeout)
        if not warm.ok:
            print("ERROR: llama-server no responde correctamente en warmup:")
            print(warm.error)
            return

    stop_event = threading.Event()
    monitor_holder = {}

    def monitor_thread_fn():
        monitor_holder["data"] = monitor_resources(stop_event)

    mon_thread = threading.Thread(target=monitor_thread_fn, daemon=True)
    mon_thread.start()

    t0 = time.perf_counter()
    results: List[RequestResult] = []
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    call_once,
                    session,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )
                for _ in range(args.requests)
            ]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

    elapsed = time.perf_counter() - t0
    stop_event.set()
    mon_thread.join(timeout=2)

    res = monitor_holder.get("data", {"cpu_samples": [], "ram_samples_mb": [], "proc_cpu_samples": [], "proc_ram_samples_mb": [], "llama_detected": False})
    summary = summarize(results, elapsed, res)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== BENCHMARK COMPLETADO ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Resultado guardado en: {args.output}")


if __name__ == "__main__":
    main()
