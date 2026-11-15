from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import ImageOnlyDataset

def extract_embeddings(model, csv_path, transform, device, batch_size=64):
    ds = ImageOnlyDataset(csv_path, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_features, paths = [], []
    with torch.no_grad():
        for x, p in loader:
            x = x.to(device)
            if hasattr(model, "embedding"):
                z = model.embedding(x)
            else:
                z = torch.flatten(model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(x)))))))), 1)
            all_features.append(z.cpu().numpy())
            paths.extend(list(p))
    return np.vstack(all_features), paths
