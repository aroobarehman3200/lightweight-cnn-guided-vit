import os
import csv
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.stop = False

    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

def top5_correct(logits, labels):
    top5 = logits.topk(5, dim=1).indices
    return top5.eq(labels.view(-1, 1)).sum().item()

def save_history_csv(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(history.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(history[keys[0]])):
            w.writerow([history[k][i] for k in keys])

def plot_curves(history, out_dir="outputs"):
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss"); plt.title("Convergence: Loss")
    plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_loss.png"), dpi=300)

    # Acc@1
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Top-1")
    plt.plot(epochs, history["val_acc"], label="Val Top-1")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Convergence: Top-1 Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_top1.png"), dpi=300)

    # Acc@5
    plt.figure()
    plt.plot(epochs, history["train_acc@5"], label="Train Top-5")
    plt.plot(epochs, history["val_acc@5"], label="Val Top-5")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Convergence: Top-5 Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_top5.png"), dpi=300)
