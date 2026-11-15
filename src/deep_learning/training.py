from pathlib import Path
import torch
import matplotlib.pyplot as plt

def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, patience, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False)
        if scheduler:
            scheduler.step(va_loss)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        if va_loss < best_loss:
            best_loss = va_loss
            wait = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_title("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(checkpoint_path.parent / "loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(history["train_acc"], label="train")
    ax.plot(history["val_acc"], label="val")
    ax.set_title("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(checkpoint_path.parent / "accuracy.png", dpi=160)
    plt.close(fig)
    return history
