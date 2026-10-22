"""Cálculos de evaluación: codo (inercia) y silueta."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def inertia_table(X_scaled, k_min: int = 1, k_max: int = 10,
                  random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        m = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        m.fit(X_scaled)
        rows.append({"K": k, "inercia": round(float(m.inertia_), 2)})
    return pd.DataFrame(rows)


def silhouette_table(X_scaled, k_min: int = 2, k_max: int = 10,
                     random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        m = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = m.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        rows.append({"K": k, "silueta": round(float(score), 4)})
    return pd.DataFrame(rows)
