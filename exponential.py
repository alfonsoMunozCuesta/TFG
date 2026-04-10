"""
Exponential survival analysis helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, ExponentialFitter


def build_exponential_analysis(df: pd.DataFrame):
    """Fit an Exponential model and build the figure and summary table."""
    if df is None or df.empty:
        return None

    required_columns = ["date", "final_result"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return None

    working_df = df[required_columns].copy()
    working_df["date"] = pd.to_numeric(working_df["date"], errors="coerce")
    working_df["final_result"] = pd.to_numeric(working_df["final_result"], errors="coerce")
    working_df = working_df.dropna(subset=["date", "final_result"])

    if working_df.empty:
        return None

    durations = working_df["date"].astype(float).where(working_df["date"] > 0, 1e-6)
    events = working_df["final_result"].astype(int)

    if durations.nunique() < 2:
        return None

    expf = ExponentialFitter()
    expf.fit(durations, event_observed=events, label="Exponential")

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="Kaplan-Meier")

    max_time = float(durations.max())
    time_grid = np.linspace(0, max_time, 200)
    fitted_survival = expf.survival_function_at_times(time_grid).values

    fitted_times = [float(value) for value in time_grid.tolist()]
    fitted_values = [float(value) for value in fitted_survival.tolist()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kmf.timeline.tolist(),
            y=kmf.survival_function_.iloc[:, 0].tolist(),
            mode="lines",
            name="Kaplan-Meier empírico",
            line=dict(color="#2c3e50", width=2, dash="dash"),
            hovertemplate="<b>Kaplan-Meier</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fitted_times,
            y=fitted_values,
            mode="lines",
            name="Curva de supervivencia exponencial",
            line=dict(color="#e67e22", width=3),
            hovertemplate="<b>Exponencial</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>",
        )
    )

    lambda_value = float(expf.lambda_)
    log_likelihood = float(expf.log_likelihood_)
    aic = float(expf.AIC_)

    fig.update_layout(
        title="Curva de supervivencia exponencial",
        xaxis_title="Tiempo",
        yaxis_title="Probabilidad de supervivencia",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=70, b=40),
        annotations=[
            dict(
                x=0.99,
                y=0.03,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                align="left",
                showarrow=False,
                bordercolor="#d0d7de",
                borderwidth=1,
                borderpad=6,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=12, color="#2c3e50"),
                text=(
                    f"<b>Valor de λ (tasa):</b> {lambda_value:.6f}<br>"
                    f"<b>Métricas</b><br>"
                    f"• Log-likelihood: {log_likelihood:.4f}<br>"
                    f"• AIC: {aic:.4f}"
                ),
            )
        ],
    )

    event_rate = (events.sum() / len(events) * 100) if len(events) else 0

    summary_df = pd.DataFrame([
        {"Metrica": "Numero de observaciones", "Valor": f"{len(events)}", "Interpretacion": "Tamanio muestral usado en el ajuste."},
        {"Metrica": "Numero de eventos", "Valor": f"{int(events.sum())}", "Interpretacion": "Casos en los que ocurre el evento de interes."},
        {"Metrica": "Tasa de eventos", "Valor": f"{event_rate:.1f}%", "Interpretacion": "Proporcion de abandonos sobre el total."},
        {"Metrica": "Lambda (tasa)", "Valor": f"{lambda_value:.6f}", "Interpretacion": "Parametro de tasa del modelo exponencial."},
        {"Metrica": "Log-likelihood", "Valor": f"{log_likelihood:.4f}", "Interpretacion": "Cuanto mayor, mejor ajuste relativo."},
        {"Metrica": "AIC", "Valor": f"{aic:.4f}", "Interpretacion": "Menor valor indica mejor equilibrio entre ajuste y complejidad."},
        {"Metrica": "Caso Weibull equivalente", "Valor": "k = 1", "Interpretacion": "El modelo exponencial es un caso particular del Weibull con forma constante."},
    ])

    interpretation = (
        f"Lambda = {lambda_value:.6f}. "
        f"Esto implica una tasa de abandono constante en el tiempo, por lo que el riesgo no cambia con el tiempo. "
        f"El modelo Exponencial es un caso particular del Weibull cuando k = 1. "
        f"La curva exponencial se compara con Kaplan-Meier empírico para ver el ajuste global."
    )

    return {
        "figure": fig,
        "summary_df": summary_df,
        "interpretation": interpretation,
        "n_observations": len(events),
        "n_events": int(events.sum()),
        "event_rate": event_rate,
        "lambda_value": lambda_value,
        "log_likelihood": log_likelihood,
        "aic": aic,
    }
