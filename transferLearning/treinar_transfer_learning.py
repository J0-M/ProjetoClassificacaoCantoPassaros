import os
import random
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as F


# =========================
# CONFIG
# =========================

pathCSV = "../data/processed/spectrograms/manifest.csv"


@dataclass
class TrainConfig:
    image_size_eff: int = 260
    image_size_res: int = 224
    batch_size: int = 32
    num_workers: int = 2
    n_splits: int = 10
    random_state: int = 42


# =========================
# SEED
# =========================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# DATASET
# =========================

class SpectrogramDataset(Dataset):
    def __init__(self, df, transform=None):
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

        extra = np.array([
            row["roi_min_freq"],
            row["roi_max_freq"],
            row["roi_duration"],
        ], dtype=np.float32)

        return image, label, extra


# =========================
# TRANSFORM (PADDING)
# =========================

class ResizeWithPadding:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        scale = min(self.size / w, self.size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = F.resize(img, (new_h, new_w))

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )

        img = F.pad(img, padding, fill=self.fill)

        return img


def build_transforms(size):
    return transforms.Compose([
        ResizeWithPadding(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =========================
# MODEL
# =========================

def build_model(model_name):
    if model_name == "efficientnet":
        weights = models.EfficientNet_B2_Weights.DEFAULT
        model = models.efficientnet_b2(weights=weights)
        model.classifier = torch.nn.Identity()
        feature_dim = 1408
        image_size = 260

    elif model_name == "resnet":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        feature_dim = 2048
        image_size = 224

    else:
        raise ValueError("Modelo inválido")

    return model, feature_dim, image_size


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(model, loader, device):
    model.eval()

    feats, labels, extras = [], [], []

    with torch.no_grad():
        for x, y, extra in loader:
            x = x.to(device)
            f = model(x)

            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
            extras.append(extra.numpy())

    return (
        np.vstack(feats),
        np.concatenate(labels),
        np.vstack(extras),
    )


# =========================
# TOP-K
# =========================

def top_k_accuracy(y_true, probs, k):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
    return correct / len(y_true)


# =========================
# FOLDS
# =========================

def create_folds(df, n_splits, seed):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    X = df["image_path"].values
    y = df["label"].values
    groups = df["audioSource"].values

    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(splitter.split(X, y, groups)):
        df.loc[val_idx, "fold"] = fold

    return df


# =========================
# TREINO POR MODELO
# =========================

def run_experiment(df, model_name, cfg):

    print(f"\n==============================")
    print(f"Modelo: {model_name}")
    print(f"==============================")

    model, feature_dim, image_size = build_model(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    transform = build_transforms(image_size)

    results = []

    for fold in range(cfg.n_splits):
        print(f"\n--- Fold {fold} ---")

        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]

        train_loader = DataLoader(
            SpectrogramDataset(train_df, transform),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        val_loader = DataLoader(
            SpectrogramDataset(val_df, transform),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        X_train, y_train, extra_train = extract_features(model, train_loader, device)
        X_val, y_val, extra_val = extract_features(model, val_loader, device)

        X_train = np.concatenate([X_train, extra_train], axis=1)
        X_val = np.concatenate([X_val, extra_val], axis=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        svm = SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            probability=True
        )

        svm.fit(X_train, y_train)

        preds = svm.predict(X_val)
        probs = svm.predict_proba(X_val)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")
        top3 = top_k_accuracy(y_val, probs, 3)
        top5 = top_k_accuracy(y_val, probs, 5)

        print(f"ACC: {acc:.4f} | F1: {f1:.4f} | Top3: {top3:.4f} | Top5: {top5:.4f}")

        results.append((f1, top3, top5))

    f1s, t3s, t5s = zip(*results)

    print("\n=== RESULTADO FINAL ===")
    print(f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Top3: {np.mean(t3s):.4f} ± {np.std(t3s):.4f}")
    print(f"Top5: {np.mean(t5s):.4f} ± {np.std(t5s):.4f}")


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "resnet", "both"]
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    seed_everything(cfg.random_state)

    df = pd.read_csv(pathCSV)
    # remove arquivos inexistentes
    df = df[df["image_path"].apply(os.path.exists)].copy()

    # remove labels nulos
    df = df.dropna(subset=["label"])

    # garante tipo consistente
    df["label"] = df["label"].astype(str)

    # remove classes raras
    min_samples = 10
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= min_samples].index
    df = df[df["label"].isin(valid_labels)]

    # encoding
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["label"])
    
    df = df.reset_index(drop=True)

    # folds
    df = create_folds(df, cfg.n_splits, cfg.random_state)

    if args.model == "both":
        for m in ["efficientnet", "resnet"]:
            run_experiment(df, m, cfg)
    else:
        run_experiment(df, args.model, cfg)


if __name__ == "__main__":
    main()