# Optimizacion Real del Modelo (Qwen2.5 + llama-server)

Esta guia resume como ajustar calidad/rendimiento en tu equipo.

## 1) Parametros de inferencia (ya soportados en START_LLAMA_SERVER.bat)

El script acepta estas variables de entorno antes de arrancar:

- N_THREADS: hilos CPU (ej. 6, 8, 12)
- N_CTX: ventana de contexto (ej. 2048, 4096)
- N_BATCH: batch de prompt (ej. 256, 512)
- N_UBATCH: micro-batch interno (ej. 64, 128)

Ejemplo en PowerShell:

```powershell
$env:N_THREADS = "8"
$env:N_CTX = "4096"
$env:N_BATCH = "512"
$env:N_UBATCH = "128"
.\START_LLAMA_SERVER.bat
```

## 2) Eleccion de cuantizacion

Regla practica (CPU-only):

- Q4_K_M: mejor equilibrio velocidad/calidad (recomendado base)
- Q5_K_M: mejor calidad, mas RAM y mas latencia
- Q6_K: calidad alta, coste de CPU/RAM mayor

Recomendacion inicial:

- RAM <= 8 GB: Q4_K_M
- RAM 12-16 GB: probar Q5_K_M
- RAM >= 24 GB: probar Q6_K

Mantener mismo prompt y mismo benchmark para comparar.

## 3) Benchmark reproducible

Se incluye script:

- benchmark_llama_server.py

Ejecutar:

```powershell
python benchmark_llama_server.py --requests 12 --concurrency 3 --max-tokens 180 --output benchmark_q4.json
```

Campos clave del resultado:

- latency_mean_s / latency_p95_s
- throughput_req_s / throughput_tok_s
- system_cpu_max_pct
- llama_cpu_max_pct
- llama_ram_max_mb

## 4) Criterios de cuello de botella

- CPU-bound: llama_cpu_max_pct cerca de 100% sostenido y latencia alta.
- RAM-bound: llama_ram_max_mb cerca del limite y/o paginacion.
- Over-batching: subir N_BATCH/N_UBATCH empeora p95.

## 5) Metodo recomendado de tuning

1. Fijar modelo (ej. Q4_K_M) y prompt.
2. Barrido de N_THREADS: 4, 6, 8, 10.
3. Con mejor threads, barrer N_BATCH/N_UBATCH.
4. Ajustar N_CTX solo si necesitas respuestas largas/contexto extenso.
5. Repetir benchmark 3 veces por configuracion y usar mediana.

## 6) Objetivo practico para dashboard

- p95 < 8 s en prompts cortos
- throughput_tok_s estable
- sin timeouts
- salida concisa y consistente en ES/EN
