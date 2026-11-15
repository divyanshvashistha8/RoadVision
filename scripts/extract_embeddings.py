from pathlib import Path
import sys, json
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader

from src.utils.config import load_config, resolve_path
from src.utils.device import get_device
from src.data.dataset import ImageOnlyDataset
from src.data.preprocessing import eval_transform
from src.deep_learning.models import build_resnet18

cfg = load_config()
device = get_device()
root = resolve_path(cfg["data"]["processed_root"])
classes = list(json.loads((root/"classes.json").read_text()).keys())

model = build_resnet18(len(classes), pretrained=False).to(device)
ckpt = torch.load(resolve_path("checkpoints/resnet18/best.pt"), map_location=device)
model.load_state_dict(ckpt["model_state"])

feature_model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
ds = ImageOnlyDataset(root/"classification/train.csv", eval_transform(cfg["data"]["image_size"]))
loader = DataLoader(ds, batch_size=cfg["embeddings"]["batch_size"], shuffle=False)

embeddings, paths = [], []
with torch.no_grad():
    for x, p in loader:
        z = feature_model(x.to(device)).flatten(1)
        embeddings.append(z.cpu().numpy())
        paths.extend(p)

emb = np.vstack(embeddings)
Path(resolve_path("outputs/embeddings")).mkdir(parents=True, exist_ok=True)
np.save(resolve_path("outputs/embeddings/train_embeddings.npy"), emb)
np.save(resolve_path("outputs/embeddings/train_paths.npy"), np.array(paths))
print("Embeddings:", emb.shape)
