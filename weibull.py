import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, WeibullFitter


def build_weibull_analysis(df):
    """Fit a Weibull model and build the figure and summary table."""
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

    wbf = WeibullFitter()
    wbf.fit(durations, event_observed=events, label="Weibull")

    max_time = float(durations.max())
    time_grid = np.linspace(0, max_time, 200)
    fitted_survival = wbf.survival_function_at_times(time_grid).values

    fitted_times = [float(value) for value in time_grid.tolist()]
    fitted_values = [float(value) for value in fitted_survival.tolist()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fitted_times,
            y=fitted_values,
            mode="lines",
            name="Weibull ajustado",
            line=dict(color="#e74c3c", width=3),
            hovertemplate="<b>Weibull</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Método Weibull",
        xaxis_title="Tiempo",
        yaxis_title="Probabilidad de supervivencia",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=70, b=40),
    )

    event_rate = (events.sum() / len(events) * 100) if len(events) else 0
    shape = float(wbf.rho_)
    scale = float(wbf.lambda_)
    median_survival = float(wbf.median_survival_time_)
    hazard_note = (
        "Aumenta con el tiempo" if shape > 1.05 else
        "Disminuye con el tiempo" if shape < 0.95 else
        "Se mantiene aproximadamente constante"
    )

    summary_df = pd.DataFrame([
        {"Metrica": "Numero de observaciones", "Valor": f"{len(events)}", "Interpretacion": "Tamanio muestral usado en el ajuste"},
        {"Metrica": "Numero de eventos", "Valor": f"{int(events.sum())}", "Interpretacion": "Casos en los que ocurre el evento"},
        {"Metrica": "Tasa de eventos", "Valor": f"{event_rate:.1f}%", "Interpretacion": "Proporcion de abandonos sobre el total"},
        {"Metrica": "Shape (rho)", "Valor": f"{shape:.4f}", "Interpretacion": f"El riesgo {hazard_note.lower()}"},
        {"Metrica": "Scale (lambda)", "Valor": f"{scale:.4f}", "Interpretacion": "Escala temporal del modelo Weibull"},
        {"Metrica": "Mediana de supervivencia", "Valor": f"{median_survival:.4f}", "Interpretacion": "Tiempo en el que la supervivencia cae al 50%"},
        {"Metrica": "Log-likelihood", "Valor": f"{float(wbf.log_likelihood_):.4f}", "Interpretacion": "Cuanto mayor, mejor ajuste relativo"},
        {"Metrica": "AIC", "Valor": f"{float(wbf.AIC_):.4f}", "Interpretacion": "Menor valor indica mejor equilibrio entre ajuste y complejidad"},
    ])

    interpretation = (
        f"Shape = {shape:.3f}. "
        f"Si es mayor que 1, el riesgo aumenta con el tiempo; si es menor que 1, disminuye. "
        f"En este ajuste, el comportamiento del riesgo es: {hazard_note.lower()}."
    )

    return {
        "figure": fig,
        "summary_df": summary_df,
        "interpretation": interpretation,
        "n_observations": len(events),
        "n_events": int(events.sum()),
        "event_rate": event_rate,
        "shape": shape,
        "scale": scale,
        "median_survival": median_survival,
        "log_likelihood": float(wbf.log_likelihood_),
        "aic": float(wbf.AIC_),
    }