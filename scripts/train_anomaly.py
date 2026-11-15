from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms

from src.utils.config import load_config, resolve_path
from src.utils.device import get_device
from src.data.dataset import ImageOnlyDataset
from src.anomaly.autoencoder import ConvAutoencoder
from src.anomaly.detector import reconstruction_scores, percentile_threshold


# Load configuration
cfg = load_config()

# Select CPU/GPU
device = get_device()

# Project data directory
root = resolve_path(cfg["data"]["processed_root"])


# Autoencoder-specific preprocessing
# Keep images in [0, 1] because the decoder uses a sigmoid output.
anomaly_transform = transforms.Compose([
    transforms.Resize(
        (cfg["data"]["image_size"], cfg["data"]["image_size"])
    ),
    transforms.ToTensor(),
])


# Load training images
ds = ImageOnlyDataset(
    root / "classification/train.csv",
    anomaly_transform
)

loader = DataLoader(
    ds,
    batch_size=cfg["data"]["batch_size"],
    shuffle=True,
    num_workers=0
)


# Create convolutional autoencoder
model = ConvAutoencoder(
    cfg["anomaly"]["latent_dim"]
).to(device)


# Optimizer and reconstruction loss
opt = optim.Adam(
    model.parameters(),
    lr=cfg["anomaly"]["learning_rate"]
)

loss_fn = nn.MSELoss()


# Train autoencoder
for epoch in range(cfg["anomaly"]["epochs"]):

    model.train()
    total = 0

    for x, _ in loader:

        x = x.to(device)

        opt.zero_grad()

        # Reconstruct the input image
        recon = model(x)

        # Reconstruction error
        loss = loss_fn(recon, x)

        # Backpropagation
        loss.backward()
        opt.step()

        total += loss.item()

    avg_loss = total / max(len(loader), 1)

    print(
        f"Epoch {epoch + 1}: "
        f"loss={avg_loss:.5f}"
    )


# Save trained model
model_path = resolve_path(
    "checkpoints/anomaly/autoencoder.pt"
)

model_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    model_path
)


# Calculate reconstruction scores
scores = reconstruction_scores(
    model,
    loader,
    device
)


# Calculate anomaly threshold
threshold = percentile_threshold(
    scores,
    cfg["anomaly"]["threshold_percentile"]
)


# Save threshold
(model_path.parent / "threshold.txt").write_text(
    str(threshold),
    encoding="utf-8"
)


print(
    "Saved anomaly model and threshold:",
    threshold
)