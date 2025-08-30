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

if __name__ == "__main__":
    main()
