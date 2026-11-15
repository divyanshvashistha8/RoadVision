from pathlib import Path
import sys, json
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config, resolve_path
from src.classical.models import train_svm, train_random_forest, save_model
from src.evaluation.classification import evaluate_classifier
from src.features.hog import extract_hog

cfg = load_config()
root = resolve_path(cfg["data"]["processed_root"])
out = resolve_path("outputs/metrics")
model_dir = resolve_path("models/classical")
out.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

def features(df):
    return np.vstack([extract_hog(p) for p in tqdm(df["image"], desc="HOG")])

train = pd.read_csv(root / "classification/train.csv")
test = pd.read_csv(root / "classification/test.csv")
classes = list(pd.read_json(root / "classes.json", typ="series").index)

Xtr, Xte = features(train), features(test)
ytr, yte = train.label.values, test.label.values

for name, model in [("svm", train_svm(Xtr, ytr)), ("random_forest", train_random_forest(Xtr, ytr))]:
    pred = model.predict(Xte)
    evaluate_classifier(yte, pred, classes, out, name)
    save_model(model, model_dir / f"{name}.joblib")
np.save(model_dir / "X_test.npy", Xte)
print("Classical models trained.")
