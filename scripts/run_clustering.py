from pathlib import Path
import sys, json, joblib
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.utils.config import resolve_path, load_config
from src.clustering.cluster import run_pca, choose_k, save_cluster_plot
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

cfg = load_config()
X = np.load(resolve_path("outputs/embeddings/train_embeddings.npy"))

pca, Z = run_pca(X, cfg["embeddings"]["pca_components"])
best_k, inertias, scores = choose_k(Z, cfg["clustering"]["min_k"], cfg["clustering"]["max_k"])
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=cfg["clustering"]["random_state"])
labels = kmeans.fit_predict(Z)

out = resolve_path("outputs/clustering")
models = resolve_path("models/clustering")
out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)

joblib.dump(pca, models/"pca.joblib")
joblib.dump(kmeans, models/"kmeans.joblib")
np.save(out/"kmeans_labels.npy", labels)
save_cluster_plot(Z[:, :2], labels, out/"kmeans.png", f"K-Means (k={best_k})")

db = DBSCAN(eps=2.0, min_samples=8).fit(Z[:, :min(20, Z.shape[1])])
db_labels = db.labels_
np.save(out/"dbscan_labels.npy", db_labels)
valid = db_labels >= 0
db_sil = float(silhouette_score(Z[valid], db_labels[valid])) if valid.sum() > 1 and len(set(db_labels[valid])) > 1 else None
print({"best_k": best_k, "silhouette": scores[best_k], "dbscan_silhouette": db_sil})
