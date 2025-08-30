import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel

from src.datasets import MiniImageNetFusionDataset
from src.models import FusionClassifier
from src.utils import EarlyStopping, top5_correct, plot_curves, save_history_csv

def main():
    # ---------- data ----------
    dataset = load_dataset("timm/mini-imagenet")
    train_ds = MiniImageNetFusionDataset(dataset["train"])
    val_ds   = MiniImageNetFusionDataset(dataset["validation"])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # ---------- models ----------
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
    resnet = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2])

    model = FusionClassifier(vit=vit, resnet=resnet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- training setup ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    early = EarlyStopping(patience=3, min_delta=0.01)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "train_acc@5": [], "val_acc@5": []}

    EPOCHS = 5
    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        tr_loss, tr_c1, tr_c5, tr_tot = 0.0, 0, 0, 0
        for x_resnet, x_vit, y in tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}", leave=False):
            x_resnet, x_vit, y = x_resnet.to(device), x_vit.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_resnet, x_vit)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_c1 += (logits.argmax(1) == y).sum().item()
            tr_c5 += top5_correct(logits, y)
            tr_tot += bs

        train_loss = tr_loss / tr_tot
        train_acc1 = 100.0 * tr_c1 / tr_tot
        train_acc5 = 100.0 * tr_c5 / tr_tot

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc1)
        history["train_acc@5"].append(train_acc5)

        # ---- val ----
        model.eval()
        va_loss, va_c1, va_c5, va_tot = 0.0, 0, 0, 0
        with torch.no_grad():
            for x_resnet, x_vit, y in tqdm(val_loader, desc="Val", leave=False):
                x_resnet, x_vit, y = x_resnet.to(device), x_vit.to(device), y.to(device)
                logits = model(x_resnet, x_vit)
                loss = criterion(logits, y)

                bs = y.size(0)
                va_loss += loss.item() * bs
                va_c1 += (logits.argmax(1) == y).sum().item()
                va_c5 += top5_correct(logits, y)
                va_tot += bs

        val_loss = va_loss / va_tot
        val_acc1 = 100.0 * va_c1 / va_tot
        val_acc5 = 100.0 * va_c5 / va_tot

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc1)
        history["val_acc@5"].append(val_acc5)

        print(f"[Epoch {epoch+1}] Train: loss {train_loss:.4f}, top1 {train_acc1:.2f}%, top5 {train_acc5:.2f}% "
              f"| Val: loss {val_loss:.4f}, top1 {val_acc1:.2f}%, top5 {val_acc5:.2f}%")

        early.step(val_acc1)
        if early.stop:
            print("Early stopping triggered.")
            break

    os.makedirs("outputs", exist_ok=True)
    save_history_csv(history, "outputs/training_history.csv")
    plot_curves(history, out_dir="outputs")

    # optional: save model weights
    torch.save(model.state_dict(), "outputs/fusion_classifier.pt")
    print("Saved: outputs/training_history.csv, plots & model in outputs/")

    # ====== TEST EVALUATION ======
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import numpy as np
    
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_resnet, x_vit, labels in tqdm(test_loader, desc="Testing"):
            x_resnet, x_vit, labels = x_resnet.to(device), x_vit.to(device), labels.to(device)
    
            # FusionClassifier returns logits directly
            outputs = model(x_resnet, x_vit)          # (B, num_classes)
    
            # Top-1
            _, top1_preds = outputs.topk(1, dim=1)
            correct_top1 += (top1_preds.squeeze(1) == labels).sum().item()
    
            # Top-5
            _, top5_preds = outputs.topk(5, dim=1)
            correct_top5 += top5_preds.eq(labels.view(-1, 1)).sum().item()
    
            # For F1, precision, recall
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
            total += labels.size(0)
    
    # Metrics
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    f1       = f1_score(all_labels, all_preds, average='weighted')
    precision= precision_score(all_labels, all_preds, average='weighted')
    recall   = recall_score(all_labels, all_preds, average='weighted')
    cm       = confusion_matrix(all_labels, all_preds)
    
    print(f"\n[TEST] Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")
    print(f"[TEST] F1 (weighted): {f1:.4f} | Precision (weighted): {precision:.4f} | Recall (weighted): {recall:.4f}")
    
    # Confusion matrix (100x100 can be dense; keep ticks off)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_test.png", dpi=300)
    
    # Optional normalized CM for readability
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title("Normalized Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_test_normalized.png", dpi=300)


if __name__ == "__main__":
    main()
