import numpy as np

def reconstruction_scores(model, loader, device):
    import torch
    model.eval()
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            err = ((x - recon) ** 2).mean(dim=(1,2,3))
            scores.extend(err.cpu().numpy())
    return np.array(scores)

def percentile_threshold(scores, percentile=95):
    return float(np.percentile(scores, percentile))
