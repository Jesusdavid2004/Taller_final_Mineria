"""Validación del esquema del dataset y compatibilidad modelo/scaler."""
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_dataset, validate_schema, EXPECTED_COLUMNS
from src.model_service import load_artifacts


def test_schema_columns():
    df = load_dataset()
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_nulls_no_duplicates():
    df = load_dataset()
    assert int(df.isnull().sum().sum()) == 0
    assert int(df.duplicated().sum()) == 0


def test_shape_and_ranges():
    df = load_dataset()
    assert df.shape == (200, 5)
    assert df["Age"].between(18, 70).all()
    assert df["Spending Score (1-100)"].between(1, 100).all()
    assert set(df["Gender"].unique()) <= {"Male", "Female"}


def test_validate_schema_clean():
    df = load_dataset()
    assert validate_schema(df) == []


def test_scaler_means_match_dataset():
    import numpy as np
    df = load_dataset()
    _, scaler = load_artifacts()
    expected = df[["Annual Income (k$)", "Spending Score (1-100)"]].mean().to_numpy()
    assert np.allclose(scaler.mean_, expected, atol=1e-6)


def test_model_matches_dataset_predictions():
    """El modelo guardado reproduce el reentrenamiento con misma semilla."""
    import numpy as np
    from sklearn.cluster import KMeans
    df = load_dataset()
    model, scaler = load_artifacts()
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    Xs = scaler.transform(X)
    ref = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xs)
    assert np.allclose(
        np.sort(model.cluster_centers_.ravel()),
        np.sort(ref.cluster_centers_.ravel()),
        atol=1e-6,
    )
