from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed
from src.data.prepare import prepare_dataset

cfg = load_config()
set_seed(cfg["project"]["seed"])

label_map = cfg["labels"]
raw_root = resolve_path(cfg["data"]["raw_root"])
out_root = resolve_path(cfg["data"]["processed_root"])

df = prepare_dataset(
    raw_root, out_root, label_map,
    seed=cfg["project"]["seed"],
    val_size=cfg["data"]["val_size"],
    test_size=cfg["data"]["test_size"],
)
print(f"Prepared {len(df)} annotated images.")
print(df["class_name"].value_counts())
