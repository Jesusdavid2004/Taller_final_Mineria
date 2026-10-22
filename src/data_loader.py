"""Carga y validación del dataset Mall Customers."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Mall_Customers.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

FEATURE_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Carga el CSV de clientes. Ruta relativa a la raíz del proyecto."""
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Devuelve una lista de problemas de esquema; vacía si todo está bien."""
    issues: list[str] = []
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            issues.append(f"Falta la columna esperada: {col}")
    if df.empty:
        issues.append("El dataset está vacío.")
    if df.isnull().sum().sum() > 0:
        issues.append("El dataset contiene valores nulos.")
    if df.duplicated().sum() > 0:
        issues.append("El dataset contiene registros duplicados.")
    return issues


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de características de entrenamiento, en orden exacto."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features: {missing}")
    return df[FEATURE_COLUMNS].copy()
