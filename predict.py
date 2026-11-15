
from pathlib import Path
import sys
import argparse
import json

sys.path.append(str(Path(__file__).resolve().parent))

import torch
from PIL import Image

from src.utils.config import load_config, resolve_path
from src.utils.device import get_device
from src.data.preprocessing import eval_transform
from src.deep_learning.models import build_resnet18
from src.detection.inference import load_yolo, predict_image
from src.severity.severity import estimate_severity


def main():
    parser = argparse.ArgumentParser(
        description="RoadVision road damage inference"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input road image"
    )

    parser.add_argument(
        "--detector",
        default="models/detection/roadvision_clean_split/weights/best.pt",
        help="Path to the trained YOLO checkpoint"
    )

    parser.add_argument(
        "--output",
        default="outputs/predictions",
        help="Directory to save annotated predictions"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    cfg = load_config()
    device = get_device()

    # Load classification metadata
    root = resolve_path(cfg["data"]["processed_root"])
    classes_path = root / "classes.json"

    if not classes_path.exists():
        raise FileNotFoundError(
            "Class mapping not found. Run the data preparation script first."
        )

    classes = list(
        json.loads(classes_path.read_text(encoding="utf-8")).keys()
    )

    # Classification
    clf_path = resolve_path("checkpoints/resnet18/best.pt")

    if clf_path.exists():
        clf = build_resnet18(
            num_classes=len(classes),
            pretrained=False
        ).to(device)

        checkpoint = torch.load(
            clf_path,
            map_location=device
        )

        state = checkpoint.get("model_state", checkpoint)
        clf.load_state_dict(state)
        clf.eval()

        transform = eval_transform(cfg["data"]["image_size"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(clf(x), dim=1)[0]

        prediction = int(probabilities.argmax())

        print("\nClassification")
        print("------------------------------")
        print(f"Predicted class: {classes[prediction]}")
        print(f"Confidence: {probabilities[prediction].item():.4f}")

    else:
        print("\nClassification checkpoint missing.")

        with Image.open(image_path) as image:
            width, height = image.size

    # Object detection
    detector_path = resolve_path(args.detector)

    if not detector_path.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {detector_path}"
        )

    detector = load_yolo(detector_path)

    detections, annotated = predict_image(
        detector,
        image_path,
        cfg["detection"]["confidence"],
        cfg["detection"]["iou"]
    )

    severity = estimate_severity(
        detections,
        width,
        height
    )

    print("\nObject Detection")
    print("------------------------------")
    print(json.dumps({
        "detections": detections,
        "severity": severity
    }, indent=2))

    # Save annotated image
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_prediction.jpg"

    import cv2

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError("Failed to save annotated prediction.")

    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
