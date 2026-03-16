import argparse
import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm


@dataclass
class SpecConfig:
    sr: Optional[int] = None
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 0
    fmax: Optional[int] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Gera espectrogramas a partir de segmentos de audio")
    parser.add_argument("--segments-csv", required=True, help="CSV com segmentos")
    parser.add_argument("--audio-root", required=True, help="Pasta raiz dos audios")
    parser.add_argument("--output-dir", required=True, help="Pasta de saida")

    parser.add_argument("--audio-col", default="soundscape_file")
    parser.add_argument("--label-col", default="roi_label")
    parser.add_argument("--start-col", default="roi_start")
    parser.add_argument("--end-col", default="roi_end")

    parser.add_argument("--jobs", type=int, default=4)

    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=None)

    parser.add_argument("--min-duration", type=float, default=0.05)
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")


def audio_segment(y: np.ndarray, sr: int, start_sec: float, end_sec: float):
    start = max(int(start_sec * sr), 0)
    end = min(int(end_sec * sr), len(y))
    if end <= start:
        return None
    segment = y[start:end]
    if segment.size == 0:
        return None
    return segment


def mel_db_image(segment: np.ndarray, sr: int, cfg: SpecConfig):
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.nan_to_num(mel_db, neginf=-80.0, posinf=0.0)

    vmin, vmax = -80.0, 0.0
    img = (np.clip((mel_db - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)
    img = np.flipud(img)
    img = np.stack([img, img, img], axis=-1)
    return img


def build_name(audio_rel: str, start: float, end: float, idx: int):
    key = f"{audio_rel}_{start:.3f}_{end:.3f}_{idx}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(audio_rel))[0]
    safe_base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base)
    return f"{safe_base}_{digest}.png"


def process_row(idx, row, args, cfg: SpecConfig, image_dir):
    label = str(row[args.label_col])
    if label in {"NOT_IDENTIFIED", "nan", "None"}:
        return None

    start = row[args.start_col]
    end = row[args.end_col]
    if pd.isna(start) or pd.isna(end):
        return None

    try:
        start = float(start)
        end = float(end)
    except ValueError:
        return None

    if end <= start:
        return None

    if (end - start) < args.min_duration:
        return None

    audio_rel = str(row[args.audio_col])
    audio_path = os.path.join(args.audio_root, audio_rel)
    if not os.path.exists(audio_path):
        return None

    try:
        y, sr = librosa.load(audio_path, sr=cfg.sr, mono=True)
        segment = audio_segment(y, sr, start, end)
        if segment is None:
            return None

        image_np = mel_db_image(segment, sr, cfg)
        img_name = build_name(audio_rel, start, end, idx)
        label_dir = os.path.join(image_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        image_path = os.path.join(label_dir, img_name)
        Image.fromarray(image_np).save(image_path, format="PNG")

        return {
            "image_path": image_path,
            "label": label,
            "audioSource": audio_rel,
            "roi_start": start,
            "roi_end": end,
            "duration": end - start,
        }
    except Exception:
        return None


def main():
    args = parse_args()
    cfg = SpecConfig(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    df = pd.read_csv(args.segments_csv)
    validate_columns(df, [args.audio_col, args.label_col, args.start_col, args.end_col])

    iterator = list(df.iterrows())
    results = Parallel(n_jobs=args.jobs)(
        delayed(process_row)(idx, row, args, cfg, image_dir) for idx, row in tqdm(iterator, desc="Gerando espectrogramas")
    )

    rows = [r for r in results if r is not None]
    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError("Nenhum espectrograma foi gerado. Verifique colunas, caminhos e filtros.")

    label_counts = manifest["label"].value_counts().sort_values(ascending=False)
    print("\nTop 10 classes:")
    print(label_counts.head(10))
    print(f"Total de imagens: {len(manifest)}")

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest salvo em: {manifest_path}")


if __name__ == "__main__":
    main()
