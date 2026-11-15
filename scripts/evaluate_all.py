from pathlib import Path
import json

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import resolve_path

metrics_dir = resolve_path("outputs/metrics")
print("Available evaluation files:")
for p in sorted(metrics_dir.glob("*.json")):
    print("-", p)
    try:
        data = json.loads(p.read_text())
        print("  accuracy:", data.get("accuracy"))
        print("  macro_f1:", data.get("macro_f1"))
    except Exception:
        pass
