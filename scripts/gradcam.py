
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# Project setup
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.device import get_device
from src.deep_learning.models import build_resnet18


CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
]


def resolve_image_path(value):
    """Resolve an image path stored in the CSV."""
    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        ROOT / path,
        resolve_path(path),
        resolve_path("data/processed") / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find image: {value}\n"
        "Check the image path stored in the validation CSV."
    )


def main():
    cfg = load_config()
    device = get_device()

    print(f"Device: {device}")

    # --------------------------------------------------
    # Load trained ResNet18 checkpoint
    # --------------------------------------------------
    checkpoint_path = resolve_path(
        "checkpoints/resnet18/best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = build_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    ).to(device)

    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # The training checkpoint contains model_state and epoch.
    if "model_state" in checkpoint_data:
        state_dict = checkpoint_data["model_state"]
    else:
        state_dict = checkpoint_data

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    # --------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------
    # Verify this normalization against train_transfer.py.
    transform = transforms.Compose([
        transforms.Resize((
            cfg["data"]["image_size"],
            cfg["data"]["image_size"],
        )),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # --------------------------------------------------
    # Select a validation image
    # --------------------------------------------------
    val_csv = resolve_path(
        "data/processed/classification/val.csv"
    )

    df = pd.read_csv(val_csv)

    print("\nValidation CSV columns:", df.columns.tolist())

    possible_columns = [
        "image_path",
        "path",
        "filepath",
        "image",
    ]

    path_column = next(
        (column for column in possible_columns
         if column in df.columns),
        None,
    )

    if path_column is None:
        raise ValueError(
            "Could not identify the image-path column. "
            f"Available columns: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError("The validation CSV contains no images.")

    # Start with the first validation image.
    image_path = resolve_image_path(df.iloc[0][path_column])

    original = Image.open(image_path).convert("RGB")
    input_tensor = transform(original).unsqueeze(0).to(device)

    # --------------------------------------------------
    # Grad-CAM hooks
    # --------------------------------------------------
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Final convolutional residual block of ResNet18.
    target_layer = model.layer4[-1]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )
    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(input_tensor)
        predicted_index = int(logits.argmax(dim=1).item())

        # Explain the predicted class.
        target_score = logits[0, predicted_index]
        target_score.backward()

        # --------------------------------------------------
        # Calculate Grad-CAM
        # --------------------------------------------------
        feature_maps = activations[0]
        grads = gradients[0]

        # Average gradients over spatial dimensions.
        channel_weights = grads.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted combination of feature maps.
        cam = (channel_weights * feature_maps).sum(
            dim=1,
            keepdim=True,
        )

        # Keep positive evidence.
        cam = F.relu(cam)

        # Resize heatmap to original image dimensions.
        cam = F.interpolate(
            cam,
            size=(original.height, original.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.detach().cpu().numpy()

        # Normalize heatmap to [0, 1].
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        # --------------------------------------------------
        # Save heatmap overlay
        # --------------------------------------------------
        output_dir = resolve_path("outputs/gradcam")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_array = np.asarray(original)

        probabilities = torch.softmax(logits, dim=1)[0]
        confidence = probabilities[predicted_index].item()

        output_path = output_dir / "resnet18_gradcam.png"

        plt.figure(figsize=(8, 8))
        plt.imshow(image_array)
        plt.imshow(cam, cmap="jet", alpha=0.4)
        plt.axis("off")
        plt.title(
            f"{CLASS_NAMES[predicted_index]} "
            f"(confidence: {confidence:.2%})"
        )
        plt.tight_layout()
        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=150,
        )
        plt.close()

        # Also save the heatmap by itself.
        heatmap_path = output_dir / "resnet18_heatmap.png"

        plt.figure(figsize=(8, 8))
        plt.imshow(cam, cmap="jet")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            heatmap_path,
            bbox_inches="tight",
            pad_inches=0,
            dpi=150,
        )
        plt.close()

        print("\n--- Grad-CAM Results ---")
        print(f"Image: {image_path}")
        print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Overlay saved to: {output_path}")
        print(f"Heatmap saved to: {heatmap_path}")

    finally:
        forward_handle.remove()
        backward_handle.remove()


if __name__ == "__main__":
    main()
