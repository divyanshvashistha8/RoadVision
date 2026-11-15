from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None):
    path = Path(path or ROOT / "configs" / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
