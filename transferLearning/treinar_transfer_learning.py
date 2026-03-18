import argparse
import copy
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


@dataclass
class TrainConfig:
    image_size: int = 224
    batch_size: int = 32
    epochs_head: int = 5
    epochs_finetune: int = 10
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    n_splits: int = 5
    random_state: int = 42


class SpectrogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label_id"])
        return image, label


def parse_args():
    parser = argparse.ArgumentParser(description="Treina transfer learning em espectrogramas")
    parser.add_argument("--manifest", required=True, help="CSV com image_path,label,audioSource")
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-class-count", type=int, default=5)

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(model_name, num_classes):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Modelo nao suportado: {model_name}")

    return model


def freeze_backbone(model, model_name):
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        for p in model.classifier.parameters():
            p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    model.train(train_mode)

    losses = []
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    return float(np.mean(losses)), float(acc), float(f1)


def create_folds(df, n_splits, seed):
    df = df.copy()

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df["image_path"].values
    y = df["label"].values
    groups = df["audioSource"].values

    fold_ids = np.full(len(df), -1)
    for fold_id, (_, val_idx) in enumerate(splitter.split(X, y, groups)):
        fold_ids[val_idx] = fold_id

    df["fold"] = fold_ids
    return df


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        n_splits=args.n_splits,
        random_state=args.seed,
    )

    seed_everything(cfg.random_state)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    df = pd.read_csv(args.manifest)
    required = {"image_path", "label", "audioSource"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest sem colunas obrigatorias: {missing}")

    df = df[df["image_path"].apply(os.path.exists)].copy()
    class_counts = df["label"].value_counts()
    valid_labels = class_counts[class_counts >= args.min_class_count].index
    df = df[df["label"].isin(valid_labels)].copy()

    if df.empty:
        raise RuntimeError("Dataset vazio apos filtros de arquivo e min-class-count")

    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["label"])
    num_classes = len(encoder.classes_)

    if num_classes < 2:
        raise RuntimeError("Numero de classes insuficiente para treino")

    if class_counts.min() < cfg.n_splits:
        print("Aviso: nem todas as classes tem suporte para os folds configurados.")

    df = create_folds(df, cfg.n_splits, cfg.random_state)

    train_tf, val_tf = build_transforms(cfg.image_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = []

    for fold in range(cfg.n_splits):
        train_df = df[df["fold"] != fold].copy()
        val_df = df[df["fold"] == fold].copy()

        train_ds = SpectrogramDataset(train_df, transform=train_tf)
        val_ds = SpectrogramDataset(val_df, transform=val_tf)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device == "cuda"),
        )

        model = build_model(args.model, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        best_f1 = -1.0
        best_state = None

        # Fase 1: treina so a cabeca
        freeze_backbone(model, args.model)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr_head,
            weight_decay=cfg.weight_decay,
        )

        for epoch in tqdm(range(cfg.epochs_head), desc=f"Fold {fold} - head"):
            tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

            history.append({
                "fold": fold,
                "phase": "head",
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_f1": tr_f1,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_f1": va_f1,
            })

            if va_f1 > best_f1:
                best_f1 = va_f1
                best_state = copy.deepcopy(model.state_dict())

        # Fase 2: fine-tuning total
        unfreeze_all(model)
        optimizer = AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

        for epoch in tqdm(range(cfg.epochs_finetune), desc=f"Fold {fold} - finetune"):
            tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

            history.append({
                "fold": fold,
                "phase": "finetune",
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_f1": tr_f1,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_f1": va_f1,
            })

            if va_f1 > best_f1:
                best_f1 = va_f1
                best_state = copy.deepcopy(model.state_dict())

        checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_fold_{fold}.pt")
        torch.save(
            {
                "model_name": args.model,
                "num_classes": num_classes,
                "classes": encoder.classes_.tolist(),
                "state_dict": best_state,
                "best_val_f1": best_f1,
                "fold": fold,
            },
            checkpoint_path,
        )
        print(f"Fold {fold} salvo em: {checkpoint_path} (best val_f1={best_f1:.4f})")

    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, "history.csv")
    history_df.to_csv(history_path, index=False)

    fold_summary = history_df.groupby("fold")["val_f1"].max().reset_index(name="best_val_f1")
    summary = {
        "model": args.model,
        "folds": int(cfg.n_splits),
        "mean_best_val_f1": float(fold_summary["best_val_f1"].mean()),
        "std_best_val_f1": float(fold_summary["best_val_f1"].std(ddof=0)),
        "num_classes": int(num_classes),
        "samples": int(len(df)),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(summary, summary_path)

    encoder_path = os.path.join(args.output_dir, "label_encoder_classes.json")
    save_json({"classes": encoder.classes_.tolist()}, encoder_path)

    print(f"History salva em: {history_path}")
    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
