
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Make the project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.device import get_device
from src.data.dataset import ImageOnlyDataset
from src.anomaly.autoencoder import ConvAutoencoder
from src.anomaly.detector import reconstruction_scores


def main():
    cfg = load_config()
    device = get_device()

    print(f"Evaluation device: {device}")

    # Must match the preprocessing used during training
    transform = transforms.Compose([
        transforms.Resize(
            (cfg["data"]["image_size"], cfg["data"]["image_size"])
        ),
        transforms.ToTensor(),
    ])

    # Use validation images, not the training images used to
    # calculate the original threshold.
    val_csv = resolve_path(
        "data/processed/classification/val.csv"
    )

    dataset = ImageOnlyDataset(val_csv, transform)

    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Load the existing trained autoencoder
    checkpoint = resolve_path(
        "checkpoints/anomaly/autoencoder.pt"
    )

    model = ConvAutoencoder(
        cfg["anomaly"]["latent_dim"]
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Calculate one reconstruction-error score per image
    scores = reconstruction_scores(model, loader, device)

    # Load the threshold calculated during training
    threshold_path = resolve_path(
        "checkpoints/anomaly/threshold.txt"
    )
    threshold = float(
        threshold_path.read_text(encoding="utf-8").strip()
    )

    flagged = scores > threshold

    print("\n--- Anomaly Evaluation Results ---")
    print(f"Validation images: {len(scores)}")
    print(f"Training-derived threshold: {threshold:.6f}")
    print(f"Mean reconstruction error: {scores.mean():.6f}")
    print(f"Median reconstruction error: {np.median(scores):.6f}")
    print(f"Standard deviation: {scores.std():.6f}")
    print(f"Minimum score: {scores.min():.6f}")
    print(f"Maximum score: {scores.max():.6f}")
    print(f"Images above threshold: {flagged.sum()}")
    print(f"Images at/below threshold: {(~flagged).sum()}")
    print(f"Flagged percentage: {flagged.mean() * 100:.2f}%")

    # Save the per-image scores for further analysis
    output_path = resolve_path(
        "outputs/metrics/anomaly_validation_scores.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        "image_index": np.arange(len(scores)),
        "reconstruction_error": scores,
        "threshold": threshold,
        "flagged": flagged,
    })

    results.to_csv(output_path, index=False)
    print(f"\nSaved scores to: {output_path}")


if __name__ == "__main__":
    main()
