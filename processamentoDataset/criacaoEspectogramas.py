import librosa
import pandas as pd
from joblib import Parallel, delayed
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import hashlib
from dataclasses import dataclass

# audioSourcePath = "../data/raw/wavs_20241104/"
# pathCSV = "../data/raw/df_ROI_RosaGLM_ConservaSom_20241104.csv"
audioSourcePath = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\wavs_20241104"
pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"

output_dir = "../data/processed/spectrograms"
image_dir = os.path.join(output_dir, "images")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)


@dataclass
class SpecConfig:
    sr: int = None
    n_fft: int = 2048
    hop_length: int = 512 # Sobreposição 75%
    n_mels: int = 128
    fmin: int = 0
    fmax: int = None


def readCSV(CSV):
    try:
        df = pd.read_csv(
            CSV,
            usecols=[
                "soundscape_file",
                "roi_label",
                "roi_start",
                "roi_end",
                "roi_label_confidence",
                "roi_min_freq",
                "roi_max_freq",
                "roi_duration",
            ],
        )
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado")
    except pd.errors.EmptyDataError:
        print("Arquivo vazio")


def audio_segment(y, sr, start, end):

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment = y[start_sample:end_sample]

    if len(segment) == 0:
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

    img = (np.clip((mel_db - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )

    img = np.flipud(img)

    img = np.stack([img, img, img], axis=-1)

    return img


def build_name(audio_rel: str, start: float, end: float, idx: int):

    key = f"{audio_rel}_{start:.3f}_{end:.3f}_{idx}".encode("utf-8")

    digest = hashlib.md5(key).hexdigest()[:12]

    base = os.path.splitext(os.path.basename(audio_rel))[0]

    safe_base = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in base
    )

    return f"{safe_base}_{digest}.png"


def process_audio_file(audio_rel, group, cfg):

    audio_path = os.path.join(audioSourcePath, audio_rel)

    if not os.path.exists(audio_path):
        return []

    try:
        y, sr = librosa.load(audio_path, sr=cfg.sr, mono=True)
    except Exception:
        return []

    results = []

    for idx, row in group.iterrows():

        label = str(row["roi_label"])

        if label == "NOT_IDENTIFIED":
            continue

        start = row["roi_start"]
        end = row["roi_end"]

        if pd.isna(start) or pd.isna(end):
            continue

        if end <= start:
            continue

        segment = audio_segment(y, sr, start, end)

        if segment is None:
            continue

        image_np = mel_db_image(segment, sr, cfg)

        img_name = build_name(audio_rel, start, end, idx)

        label_dir = os.path.join(image_dir, label)

        os.makedirs(label_dir, exist_ok=True)

        image_path = os.path.join(label_dir, img_name)

        Image.fromarray(image_np).save(image_path, format="PNG")

        results.append(
            {
                "image_path": image_path,
                "label": label,
                "audioSource": audio_rel,
                "roi_start": start,
                "roi_end": end,
                "roi_min_freq": row["roi_min_freq"],
                "roi_max_freq": row["roi_max_freq"],
                "roi_duration": row["roi_duration"],
            }
        )

    return results


def main():

    cfg = SpecConfig()

    df = readCSV(pathCSV)

    if df is None:
        print("Falha ao carregar o CSV.")
        return

    print("Total de linhas:", len(df))

    grouped = df.groupby("soundscape_file", sort=False)

    results = Parallel(n_jobs=4)(
        delayed(process_audio_file)(audio_file, group, cfg)
        for audio_file, group in tqdm(grouped)
    )

    results = [item for sublist in results for item in sublist]

    manifest = pd.DataFrame(results)

    manifest_path = os.path.join(output_dir, "manifest.csv")

    manifest.to_csv(manifest_path, index=False)

    print("Total de espectrogramas gerados:", len(manifest))


if __name__ == "__main__":
    main()