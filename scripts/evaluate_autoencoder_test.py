
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.anomaly.autoencoder import ConvAutoencoder


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "data/processed/classification/test.csv"
CHECKPOINT_PATH = ROOT / "checkpoints/anomaly/autoencoder.pt"
THRESHOLD_PATH = ROOT / "checkpoints/anomaly/threshold.txt"

OUTPUT_DIR = ROOT / "outputs/anomaly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Load test dataset
# --------------------------------------------------

class RoadDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(row["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image


# --------------------------------------------------
# 3. Load trained autoencoder
# --------------------------------------------------

model = ConvAutoencoder(latent_dim=128).to(DEVICE)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

threshold = float(THRESHOLD_PATH.read_text().strip())

print("Device:", DEVICE)
print("Threshold:", threshold)


# --------------------------------------------------
# 4. Load test CSV
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

dataset = RoadDataset(df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(dataset))


# --------------------------------------------------
# 5. Calculate reconstruction errors
# --------------------------------------------------

all_scores = []

with torch.no_grad():

    for batch in loader:

        batch = batch.to(DEVICE)

        reconstructed = model(batch)

        errors = ((batch - reconstructed) ** 2).mean(
            dim=(1, 2, 3)
        )

        all_scores.extend(errors.cpu().numpy())

scores = np.array(all_scores)


# --------------------------------------------------
# 6. Compare against threshold
# --------------------------------------------------

results = df.copy()

results["reconstruction_error"] = scores

results["above_threshold"] = (
    results["reconstruction_error"] > threshold
)

results["threshold"] = threshold

results.to_csv(
    OUTPUT_DIR / "test_reconstruction_scores.csv",
    index=False
)


# --------------------------------------------------
# 7. Overall statistics
# --------------------------------------------------

print("\n========== TEST RESULTS ==========")

print("Total images:", len(scores))
print("Mean error:", round(float(scores.mean()), 6))
print("Median error:", round(float(np.median(scores)), 6))
print("Standard deviation:", round(float(scores.std()), 6))
print("Minimum error:", round(float(scores.min()), 6))
print("Maximum error:", round(float(scores.max()), 6))

above = int((scores > threshold).sum())

print("\nThreshold:", threshold)
print("Above threshold:", above)
print("Below threshold:", len(scores) - above)

print(
    "Percentage above threshold:",
    round(100 * above / len(scores), 2),
    "%"
)


# --------------------------------------------------
# 8. Class-wise reconstruction analysis
# --------------------------------------------------

print("\n========== CLASS-WISE RESULTS ==========")

summary = results.groupby("class_name").agg(
    image_count=("image", "count"),
    mean_error=("reconstruction_error", "mean"),
    median_error=("reconstruction_error", "median"),
    max_error=("reconstruction_error", "max"),
    above_threshold=("above_threshold", "sum")
)

print(summary.round(6).to_string())

summary.to_csv(
    OUTPUT_DIR / "classwise_anomaly_summary.csv"
)

print("\nEvaluation completed successfully.")
print("Reports saved in:", OUTPUT_DIR)
