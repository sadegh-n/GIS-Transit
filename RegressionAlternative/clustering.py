"""
Clustering analysis for neighborhood typology.
Groups block groups by built environment characteristics to identify
what kinds of areas are underserved by transit.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

FEATURES = [
    "D1B",          # population density
    "D1C",          # employment density
    "D3B",          # intersection density (walkability)
    "NatWalkInd",   # national walkability index
    "D4A",          # distance to nearest transit stop
    "D4C",          # transit service frequency
    "Pct_AO0",      # zero-car household share
    "R_PCTLOWWAGE", # low-wage worker share
    "transit_ratio", # transit vs auto job access
    "D2A_JPHH",     # jobs-housing balance
]

SKEWED = ["D1B", "D1C", "D3B", "D4A", "D4C"]


def prepare_features(county_df):
    """Standardize and transform features for clustering."""
    X = county_df[FEATURES].fillna(county_df[FEATURES].median()).copy()
    for col in SKEWED:
        X[col] = np.log1p(X[col])
    return StandardScaler().fit_transform(X)


def find_best_k(X, k_range=range(3, 9)):
    """Test multiple k values and pick the one with the best silhouette score."""
    best_k, best_score = 5, -1
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        s = silhouette_score(X, lab)
        scores[k] = round(s, 4)
        if s > best_score:
            best_k, best_score = k, s
    return best_k, scores


def run_clustering(county_df):
    """
    Choose k via silhouette score, run K-means, and compare against
    hierarchical clustering and DBSCAN.
    """
    X = prepare_features(county_df)

    # Pick k automatically
    best_k, k_scores = find_best_k(X)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    metrics = {}
    # Report silhouette for each k tested
    for k, s in k_scores.items():
        suffix = " *" if k == best_k else ""
        metrics[f"K-Means (k={k}){suffix}"] = {
            "Silhouette": s,
        }

    # Add CH score for selected k
    metrics[f"K-Means (k={best_k}) *"]["Calinski-Harabasz"] = calinski_harabasz_score(X, labels)

    # Comparison methods using same k
    hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_labels = hc.fit_predict(X)
    metrics[f"Hierarchical (k={best_k})"] = {
        "Silhouette": silhouette_score(X, hc_labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, hc_labels),
    }

    db = DBSCAN(eps=1.2, min_samples=10)
    db_labels = db.fit_predict(X)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_clusters >= 2:
        mask = db_labels != -1
        metrics["DBSCAN"] = {
            "Silhouette": silhouette_score(X[mask], db_labels[mask]),
            "Calinski-Harabasz": calinski_harabasz_score(X[mask], db_labels[mask]),
        }

    return labels, metrics


def profile_clusters(county_df, labels):
    """
    Compute mean feature values per cluster for comparison.
    Does not assign names — clusters are numbered and described by their data.
    """
    df = county_df.copy()
    df["cluster"] = labels

    profile_cols = FEATURES + ["TotPop", "TotEmp"]
    available = [c for c in profile_cols if c in df.columns]
    profiles = df.groupby("cluster")[available].mean()
    profiles["count"] = df.groupby("cluster").size()

    return profiles
