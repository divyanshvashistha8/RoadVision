from pathlib import Path
import sys, json
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.dataset import ImageClassificationDataset
from src.data.preprocessing import train_transform, eval_transform
from src.deep_learning.models import RoadCNN
from src.deep_learning.training import fit
from src.evaluation.classification import evaluate_classifier

cfg = load_config()
set_seed(cfg["project"]["seed"])
device = get_device()
root = resolve_path(cfg["data"]["processed_root"])
train_ds = ImageClassificationDataset(root/"classification/train.csv", train_transform(cfg["data"]["image_size"]))
val_ds = ImageClassificationDataset(root/"classification/val.csv", eval_transform(cfg["data"]["image_size"]))
test_ds = ImageClassificationDataset(root/"classification/test.csv", eval_transform(cfg["data"]["image_size"]))

train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=device.type=="cuda")
val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

classes = list(json.loads((root/"classes.json").read_text()).keys())
model = RoadCNN(len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg["classification"]["learning_rate"], weight_decay=cfg["classification"]["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    cfg["classification"]["epochs"], cfg["classification"]["patience"], resolve_path("checkpoints/cnn/best.pt"))

ckpt = torch.load(resolve_path("checkpoints/cnn/best.pt"), map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
ys, ps = [], []
with torch.no_grad():
    for x, y, _ in test_loader:
        pred = model(x.to(device)).argmax(1).cpu()
        ys.extend(y.tolist()); ps.extend(pred.tolist())
evaluate_classifier(ys, ps, classes, resolve_path("outputs/metrics"), "cnn")
print("CNN training complete on", device)
