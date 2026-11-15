
from pathlib import Path
import sys

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path


CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
]


def estimate_visual_extent(area_percent):
    """
    Heuristic based on the fraction of the image covered
    by an individual predicted bounding box.

    These are provisional thresholds, not engineering standards.
    """
    if area_percent < 1.0:
        return "Low visual extent"
    elif area_percent < 5.0:
        return "Moderate visual extent"
    return "High visual extent"


def main():
    checkpoint = resolve_path(
        "models/detection/roadvision_clean_split/weights/best.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {checkpoint}"
        )

    model = YOLO(str(checkpoint))

    image_input = input(
        "Enter the road image path: "
    ).strip().strip('"')

    image_path = Path(image_input)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Run object detection.
    results = model.predict(
        source=str(image_path),
        imgsz=416,
        conf=0.25,
        save=False,
        verbose=False,
    )

    result = results[0]
    image_height, image_width = result.orig_shape
    image_area = image_height * image_width

    rows = []

    if result.boxes is not None:
        boxes = result.boxes

        for index in range(len(boxes)):
            x1, y1, x2, y2 = (
                boxes.xyxy[index].cpu().tolist()
            )

            class_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())

            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            box_area = box_width * box_height

            area_percent = 100.0 * box_area / image_area

            if class_id < len(CLASS_NAMES):
                damage_class = CLASS_NAMES[class_id]
            else:
                damage_class = model.names[class_id]

            rows.append({
                "detection_id": index + 1,
                "damage_class": damage_class,
                "confidence": round(confidence, 4),
                "bbox_x1": round(x1, 2),
                "bbox_y1": round(y1, 2),
                "bbox_x2": round(x2, 2),
                "bbox_y2": round(y2, 2),
                "bbox_area_percent": round(area_percent, 4),
                "visual_extent": estimate_visual_extent(
                    area_percent
                ),
            })

    output_dir = resolve_path("outputs/severity")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save an annotated image.
    annotated = result.plot()
    annotated_path = output_dir / "severity_annotated.jpg"

    import cv2
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RuntimeError(
            f"Could not save annotated image: {annotated_path}"
        )

    # Save the detection report.
    report_path = output_dir / "severity_report.csv"

    columns = [
        "detection_id",
        "damage_class",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_area_percent",
        "visual_extent",
    ]

    report = pd.DataFrame(rows, columns=columns)
    report.to_csv(report_path, index=False)

    print("\n--- RoadVision Visual Extent Report ---")
    print(f"Image: {image_path}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Damage detections: {len(rows)}")

    if rows:
        print("\nDetections:")
        print(report[
            [
                "damage_class",
                "confidence",
                "bbox_area_percent",
                "visual_extent",
            ]
        ].to_string(index=False))

        print(
            "\nNote: categories describe bounding-box extent only. "
            "They do not establish structural severity."
        )
    else:
        print(
            "No objects passed the confidence threshold. "
            "This does not prove the road is damage-free."
        )

    print(f"\nAnnotated image: {annotated_path}")
    print(f"CSV report: {report_path}")


if __name__ == "__main__":
    main()