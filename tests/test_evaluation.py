"""Regresión de la evaluación: codo y silueta con valores verificados."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from src.data_loader import load_dataset, get_feature_matrix
from src.model_service import load_artifacts
from src.analysis import inertia_table, silhouette_table

EXPECTED_INERTIA = {
    1: 400.00, 2: 269.69, 3: 157.70, 4: 108.92, 5: 65.57,
    6: 55.06, 7: 44.86, 8: 37.23, 9: 32.39, 10: 29.98,
}
EXPECTED_SIL = {
    2: 0.3213, 3: 0.4666, 4: 0.4939, 5: 0.5547, 6: 0.5399,
    7: 0.5281, 8: 0.4552, 9: 0.4571, 10: 0.4432,
}


@pytest.fixture(scope="module")
def scaled():
    df = load_dataset()
    _, scaler = load_artifacts()
    return scaler.transform(get_feature_matrix(df))


def test_inertia_decreases_and_matches(scaled):
    t = inertia_table(scaled)
    assert list(t["K"]) == list(range(1, 11))
    vals = list(t["inercia"])
    assert all(b < a for a, b in zip(vals, vals[1:])), "La inercia debe decrecer con K"
    for _, row in t.iterrows():
        assert row["inercia"] == pytest.approx(EXPECTED_INERTIA[int(row["K"])], abs=0.05)


def test_silhouette_max_at_k5(scaled):
    t = silhouette_table(scaled)
    assert list(t["K"]) == list(range(2, 11))
    for _, row in t.iterrows():
        assert row["silueta"] == pytest.approx(EXPECTED_SIL[int(row["K"])], abs=0.002)
    best = t.loc[t["silueta"].idxmax()]
    assert int(best["K"]) == 5
    assert float(best["silueta"]) == pytest.approx(0.5547, abs=0.002)
