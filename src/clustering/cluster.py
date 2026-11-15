from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def run_pca(X, n_components=50):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    return pca, Z

def choose_k(Z, min_k=2, max_k=10):
    max_k = min(max_k, len(Z)-1)
    scores, inertias = {}, {}
    for k in range(min_k, max_k+1):
        m = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = m.fit_predict(Z)
        inertias[k] = float(m.inertia_)
        scores[k] = float(silhouette_score(Z, labels))
    best_k = max(scores, key=scores.get)
    return best_k, inertias, scores

def save_cluster_plot(Z2, labels, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z2[:,0], Z2[:,1], c=labels, s=12)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
