"""
Random Survival Forest helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def _coerce_event_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    positive_values = {"1", "true", "yes", "y", "withdrawn", "dropout", "abandono", "event"}
    return normalized.isin(positive_values).astype(int)


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    if "date" not in df.columns or "final_result" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    working_df = df.copy()
    working_df["date"] = pd.to_numeric(working_df["date"], errors="coerce")
    working_df["final_result"] = _coerce_event_column(working_df["final_result"])
    working_df = working_df.dropna(subset=["date", "final_result"])

    if working_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    durations = working_df["date"].astype(float).clip(lower=1e-6)
    events = working_df["final_result"].astype(int)

    feature_df = working_df.drop(columns=["date", "final_result"], errors="ignore")
    feature_df = feature_df.select_dtypes(include=["number", "bool", "object", "category"])
    feature_df = pd.get_dummies(feature_df, drop_first=False)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    if feature_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True)).fillna(0.0)

    valid_rows = durations.notna() & events.notna()
    feature_df = feature_df.loc[valid_rows].reset_index(drop=True)
    durations = durations.loc[valid_rows].reset_index(drop=True)
    events = events.loc[valid_rows].reset_index(drop=True)

    return feature_df, durations, events


def _build_importance_proxy(feature_df: pd.DataFrame, risk_scores: np.ndarray) -> np.ndarray:
    correlations = []
    for column in feature_df.columns:
        series = pd.to_numeric(feature_df[column], errors="coerce").fillna(feature_df[column].median())
        if series.nunique(dropna=True) <= 1:
            correlations.append(0.0)
            continue
        try:
            value = np.corrcoef(series.to_numpy(dtype=float), risk_scores)[0, 1]
            correlations.append(float(abs(value)) if np.isfinite(value) else 0.0)
        except Exception:
            correlations.append(0.0)
    return np.asarray(correlations, dtype=float)


def build_rsf_analysis(df: pd.DataFrame):
    """Fit a Random Survival Forest and build reusable outputs for the dashboard."""

    feature_df, durations, events = _build_feature_matrix(df)
    if feature_df.empty or durations.nunique() < 2:
        return None

    if len(feature_df) < 5 or feature_df.shape[1] < 1:
        return None

    y = Surv.from_arrays(event=events.astype(bool).to_numpy(), time=durations.to_numpy(dtype=float))

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )
    rsf.fit(feature_df, y)

    risk_scores = rsf.predict(feature_df)
    train_c_index = concordance_index_censored(events.astype(bool).to_numpy(), durations.to_numpy(dtype=float), risk_scores)[0]
    oob_score = getattr(rsf, "oob_score_", None)

    sorted_indices = np.argsort(risk_scores)
    if len(sorted_indices) == 0:
        return None

    percentile_positions = [0.2, 0.5, 0.8]
    representative_indices = []
    for percentile in percentile_positions:
        position = int(round((len(sorted_indices) - 1) * percentile))
        candidate = int(sorted_indices[position])
        while candidate in representative_indices and position + 1 < len(sorted_indices):
            position += 1
            candidate = int(sorted_indices[position])
        representative_indices.append(candidate)

    survival_curves = rsf.predict_survival_function(feature_df.iloc[representative_indices], return_array=True)
    curve_times = np.asarray(rsf.unique_times_, dtype=float)

    if len(feature_df.columns) > 0:
        try:
            importances = rsf.feature_importances_
        except (AttributeError, NotImplementedError):
            importances = None

        if importances is None or not np.isfinite(importances).any() or np.allclose(importances, 0):
            importances = _build_importance_proxy(feature_df, risk_scores)
    else:
        importances = np.asarray([])

    importances = np.asarray(importances, dtype=float)
    if importances.size:
        normalized_importances = importances / importances.sum() if importances.sum() > 0 else importances
        top_indices = np.argsort(normalized_importances)[::-1][:10]
        top_features = [
            {
                "name": feature_df.columns[index],
                "importance": float(normalized_importances[index]),
            }
            for index in top_indices
            if normalized_importances[index] > 0
        ]
    else:
        top_features = []

    top_feature_name = top_features[0]["name"] if top_features else "N/A"
    top_feature_value = top_features[0]["importance"] if top_features else 0.0

    survival_labels = ["Bajo riesgo", "Riesgo medio", "Alto riesgo"]
    if len(representative_indices) < 3:
        survival_labels = [f"Grupo {index + 1}" for index in range(len(representative_indices))]

    fig_survival = go.Figure()
    colors = ["#1abc9c", "#2980b9", "#e74c3c"]
    y_min = 1.0

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)
        return f"rgba({red}, {green}, {blue}, {alpha})"

    for index, row_index in enumerate(representative_indices):
        curve = survival_curves[index]
        y_min = min(y_min, float(np.min(curve)))
        fig_survival.add_trace(
            go.Scatter(
                x=curve_times,
                y=curve,
                mode="lines+markers",
                name=f"{survival_labels[index]} (score={risk_scores[row_index]:.3f})",
                line=dict(color=colors[index % len(colors)], width=4),
                line_shape="hv",
                marker=dict(size=5, color=colors[index % len(colors)]),
                hovertemplate="<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>",
                showlegend=True,
            )
        )

    x_axis_max = float(np.max(curve_times)) if len(curve_times) else 1.0
    if x_axis_max <= 0:
        x_axis_max = 1.0

    y_axis_min = max(0.70, y_min - 0.08)
    if y_axis_min > 0.92:
        y_axis_min = 0.92

    fig_survival.update_layout(
        title="Random Survival Forest: curvas de supervivencia estimadas",
        xaxis_title="Tiempo",
        yaxis_title="Probabilidad de supervivencia",
        xaxis=dict(range=[0, x_axis_max], fixedrange=True),
        yaxis=dict(range=[y_axis_min, 1], fixedrange=True),
        height=720,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=60),
    )

    importance_df = pd.DataFrame(top_features)
    if not importance_df.empty:
        importance_fig = go.Figure(
            go.Bar(
                x=importance_df["importance"].iloc[::-1],
                y=importance_df["name"].iloc[::-1],
                orientation="h",
                marker=dict(color="#2980b9"),
                hovertemplate="<b>%{y}</b><br>Importancia: %{x:.3f}<extra></extra>",
            )
        )
    else:
        importance_fig = go.Figure()

    importance_fig.update_layout(
        title="Importancia de variables en RSF",
        xaxis_title="Importancia normalizada",
        yaxis_title="Variable",
        template="plotly_white",
        margin=dict(l=180, r=30, t=70, b=40),
        height=max(460, 60 * max(len(top_features), 3)),
    )

    summary_df = pd.DataFrame([
        {
            "Metrica": "Observaciones",
            "Valor": f"{len(feature_df)}",
            "Interpretacion": "Número de filas usadas para entrenar el bosque.",
        },
        {
            "Metrica": "Eventos",
            "Valor": f"{int(events.sum())}",
            "Interpretacion": "Casos en los que se produjo el evento de interés.",
        },
        {
            "Metrica": "Variables",
            "Valor": f"{feature_df.shape[1]}",
            "Interpretacion": "Número de predictores usados tras el preprocesado.",
        },
        {
            "Metrica": "Concordancia train",
            "Valor": f"{train_c_index:.3f}",
            "Interpretacion": "Cuanto más cerca de 1, mejor ordena los riesgos.",
        },
        {
            "Metrica": "Concordancia OOB",
            "Valor": f"{oob_score:.3f}" if oob_score is not None else "N/A",
            "Interpretacion": "Estimación fuera de bolsa del rendimiento general.",
        },
        {
            "Metrica": "Variable más influyente",
            "Valor": top_feature_name,
            "Interpretacion": f"Peso relativo aproximado: {top_feature_value:.3f}",
        },
    ])

    if oob_score is not None:
        interpretation = (
            f"El Random Survival Forest ha combinado {feature_df.shape[1]} variables mediante árboles entrenados sobre muestras bootstrap. "
            f"La concordancia en entrenamiento es {train_c_index:.3f} y la estimación OOB es {oob_score:.3f}. "
            f"La variable más influyente es {top_feature_name}. Las curvas muestran cómo cambia la supervivencia para perfiles de bajo, medio y alto riesgo."
        )
    else:
        interpretation = (
            f"El Random Survival Forest ha combinado {feature_df.shape[1]} variables mediante árboles entrenados sobre muestras bootstrap. "
            f"La concordancia en entrenamiento es {train_c_index:.3f}. La variable más influyente es {top_feature_name}. "
            f"Las curvas muestran cómo cambia la supervivencia para perfiles de bajo, medio y alto riesgo."
        )

    return {
        "figure": fig_survival,
        "importance_figure": importance_fig,
        "summary_df": summary_df,
        "interpretation": interpretation,
        "n_observations": len(feature_df),
        "n_events": int(events.sum()),
        "n_features": int(feature_df.shape[1]),
        "train_c_index": float(train_c_index),
        "oob_score": None if oob_score is None else float(oob_score),
        "top_feature": top_feature_name,
        "top_feature_importance": float(top_feature_value),
        "top_features": top_features,
        "survival_labels": survival_labels,
    }