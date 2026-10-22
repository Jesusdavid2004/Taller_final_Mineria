"""Tests de inferencia con artefactos reales guardados."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from src.model_service import (
    load_artifacts, check_compatibility, predict_cluster,
    validate_customer_input, centroids_original_units, cluster_summary,
    FEATURE_COLUMNS, N_CLUSTERS,
)
from src.data_loader import load_dataset, get_feature_matrix
from src.business_insights import CLUSTER_PROFILES, get_profile


def test_artifacts_load():
    model, scaler = load_artifacts()
    assert model.n_clusters == N_CLUSTERS
    assert list(scaler.feature_names_in_) == FEATURE_COLUMNS


def test_compatibility_ok():
    model, scaler = load_artifacts()
    assert check_compatibility(model, scaler) == []


def test_feature_order_and_dim():
    _, scaler = load_artifacts()
    assert scaler.n_features_in_ == 2
    assert list(scaler.feature_names_in_) == ["Annual Income (k$)", "Spending Score (1-100)"]


def test_valid_prediction_returns_known_cluster():
    model, scaler = load_artifacts()
    label = predict_cluster(60, 50, model, scaler)
    assert label in range(N_CLUSTERS)


def test_prediction_deterministic():
    model, scaler = load_artifacts()
    assert predict_cluster(86, 82, model, scaler) == predict_cluster(86, 82, model, scaler)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        validate_customer_input("abc", 50)
    with pytest.raises(ValueError):
        validate_customer_input(60, 500)
    with pytest.raises(ValueError):
        validate_customer_input(-5, 50)
    with pytest.raises(ValueError):
        validate_customer_input(None, None)


def test_centroids_shape_and_units():
    model, scaler = load_artifacts()
    c = centroids_original_units(model, scaler)
    assert c.shape == (5, 2)
    assert list(c.columns) == FEATURE_COLUMNS
    assert (c["Annual Income (k$)"] > 0).all()
    assert ((c["Spending Score (1-100)"] >= 1) & (c["Spending Score (1-100)"] <= 100)).all()


def test_cluster_summary_counts():
    df = load_dataset()
    model, scaler = load_artifacts()
    X = get_feature_matrix(df)
    labels = model.predict(scaler.transform(X))
    s = cluster_summary(df, labels)
    assert s["n"].sum() == len(df)
    assert set(s.index) == set(range(N_CLUSTERS))
    assert abs(s["pct"].sum() - 100.0) < 0.2


def test_profiles_cover_all_clusters():
    for k in range(N_CLUSTERS):
        p = get_profile(k)
        assert {"nombre", "estrategia", "oportunidad", "riesgo"} <= set(p)
    with pytest.raises(ValueError):
        get_profile(99)
