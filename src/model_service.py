"""Servicio de inferencia: scaler + K-Means guardados (sin reentrenar en predicción)."""
from __future__ import annotations

import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "kmeans_model.pkl"
SCALER_PATH = PROJECT_ROOT / "model" / "scaler.pkl"

FEATURE_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]
N_CLUSTERS = 5


def load_artifacts(model_path: Path | str = MODEL_PATH,
                   scaler_path: Path | str = SCALER_PATH):
    """Carga (modelo K-Means, scaler). Ignora el warning de versión si existe."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    return model, scaler


def check_compatibility(model, scaler) -> list[str]:
    """Verifica que modelo y scaler sean mutuamente compatibles."""
    issues: list[str] = []
    if getattr(model, "n_clusters", None) != N_CLUSTERS:
        issues.append(f"El modelo tiene {getattr(model, 'n_clusters', '?')} clusters, se esperaban {N_CLUSTERS}.")
    feat = getattr(scaler, "feature_names_in_", None)
    feat = list(feat) if feat is not None else []
    if feat != FEATURE_COLUMNS:
        issues.append(f"Features del scaler: {feat}; esperadas: {FEATURE_COLUMNS}.")
    if getattr(scaler, "n_features_in_", None) != len(FEATURE_COLUMNS):
        issues.append("Dimensión del scaler incompatible con las features.")
    if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
        issues.append("Dimensión del modelo incompatible con las features.")
    return issues


def validate_customer_input(income, spending) -> tuple[float, float]:
    """Valida el formulario de nuevo cliente. Lanza ValueError con mensaje en español."""
    try:
        income_f = float(income)
    except (TypeError, ValueError):
        raise ValueError("El ingreso anual debe ser un número válido.")
    try:
        spending_f = float(spending)
    except (TypeError, ValueError):
        raise ValueError("El spending score debe ser un número válido (1-100).")
    if not 0 <= income_f <= 200:
        raise ValueError("El ingreso anual debe estar entre 0 y 200 (miles de USD).")
    if not 1 <= spending_f <= 100:
        raise ValueError("El spending score debe estar entre 1 y 100.")
    return income_f, spending_f


def predict_cluster(income: float, spending: float, model=None, scaler=None) -> int:
    """Clasifica un cliente nuevo. NO reentrena el scaler: solo usa transform()."""
    income_f, spending_f = validate_customer_input(income, spending)
    if model is None or scaler is None:
        model, scaler = load_artifacts()
    x = pd.DataFrame([[income_f, spending_f]], columns=FEATURE_COLUMNS)
    scaled = scaler.transform(x)
    label = int(model.predict(scaled)[0])
    if label not in range(N_CLUSTERS):
        raise ValueError(f"Etiqueta de cluster inválida: {label}")
    return label


def centroids_original_units(model=None, scaler=None) -> pd.DataFrame:
    """Centroides en unidades originales (k$ y score 1-100)."""
    if model is None or scaler is None:
        model, scaler = load_artifacts()
    data = scaler.inverse_transform(model.cluster_centers_)
    return pd.DataFrame(data, columns=FEATURE_COLUMNS).round(2)


def cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Resumen por cluster: n, %, edad media/mediana, ingreso y gasto medios."""
    tmp = df.copy()
    tmp["Cluster"] = labels
    summary = tmp.groupby("Cluster").agg(
        n=("CustomerID", "count"),
        edad_media=("Age", "mean"),
        edad_mediana=("Age", "median"),
        ingreso_medio=("Annual Income (k$)", "mean"),
        gasto_medio=("Spending Score (1-100)", "mean"),
    ).round(2)
    summary["pct"] = (summary["n"] / len(tmp) * 100).round(1)
    return summary.sort_index()
