from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from src.utils.config import resolve_path

# Corrected dataset configuration
data_yaml = resolve_path("data/processed/yolo/data.yaml")

# Start from pretrained YOLO11n weights.
# Do not reuse the old fine-tuned checkpoint.
model = YOLO("yolo11n.pt")

model.train(
    data=str(data_yaml),
    epochs=10,
    imgsz=416,
    batch=4,
    patience=5,
    project=str(resolve_path("models/detection")),
    name="roadvision_clean_split",
)

print("Clean-split YOLO training complete.")
print("Results: models/detection/roadvision_clean_split/")