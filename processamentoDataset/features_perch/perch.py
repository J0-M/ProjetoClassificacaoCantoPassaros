import os, gc, warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import pickle
import onnxruntime as ort
from tqdm.auto import tqdm

DATA_VERSION = "v6_perch"

audioSourcePath = r"D:\Users\Joao\Downloads\dados_RosaGLM_ConservaSom_20241104\wavs_20241104"
pathCSV         = r"D:\Users\Joao\Downloads\dados_RosaGLM_ConservaSom_20241104\df_ROI_RosaGLM_ConservaSom_20241104.csv"

HERE = Path(__file__).resolve().parent
PERCH_ONNX_PATH = HERE / "perch_v2_no_dft.onnx"

if not PERCH_ONNX_PATH.exists():
    raise FileNotFoundError(f"ONNX não encontrado: {PERCH_ONNX_PATH}")

SR_PERCH       = 32000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR_PERCH * WINDOW_SEC
BATCH_WINDOWS  = 32
AGG_MODE       = "mean"

OUTPUT_DIR = Path(f"../../dataframes/{DATA_VERSION}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PKL = OUTPUT_DIR / "dataframeSegmentado.pkl"


def readCSV(CSV):
    try:
        return pd.read_csv(CSV, usecols=["soundscape_file", "roi_label",
                                          "roi_start", "roi_end",
                                          "roi_label_confidence",
                                          "roi_min_freq", "roi_max_freq",
                                          "roi_duration"])
    except FileNotFoundError:
        print("Arquivo não encontrado")
    except pd.errors.EmptyDataError:
        print("Arquivo Vazio")
    return None


def filter_df(df):
    mask = (
        (df["roi_label"] != "NOT_IDENTIFIED")
        & (df["roi_label_confidence"] != "uncertain")
        & df["roi_min_freq"].notna()
        & df["roi_max_freq"].notna()
        & df["roi_duration"].notna()
    )
    return df[mask].reset_index(drop=True)


def load_segment(audio_rel, start, end):
    if pd.isna(start) or pd.isna(end) or start >= end:
        return None
    full = os.path.join(audioSourcePath, audio_rel)
    if not os.path.exists(full):
        return None
    try:
        duration = float(end) - float(start)
        y, _ = librosa.load(full, sr=SR_PERCH, mono=True,
                            offset=float(start), duration=duration)
        if y.size == 0:
            return None
        return y.astype(np.float32)
    except Exception as e:
        print(f"Erro carregando {audio_rel}: {e}")
        return None


def segment_to_windows(seg):
    if seg.size <= WINDOW_SAMPLES:
        return [np.pad(seg, (0, WINDOW_SAMPLES - seg.size))]

    wins = []
    for s in range(0, seg.size - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        wins.append(seg[s: s + WINDOW_SAMPLES])

    remainder = seg.size % WINDOW_SAMPLES
    if remainder > 0:
        tail = seg[-remainder:]
        wins.append(np.pad(tail, (0, WINDOW_SAMPLES - remainder)))

    return wins


def aggregate_windows(embs, logits, mode="mean"):
    if mode == "mean":
        return np.concatenate([embs.mean(0), logits.mean(0)])
    if mode == "max":
        return np.concatenate([embs.max(0), logits.max(0)])
    if mode == "mean_std":
        return np.concatenate([embs.mean(0), embs.std(0),
                               logits.mean(0), logits.max(0)])
    raise ValueError(mode)


def infer_windows(sess, in_name, out_map, wins):
    embs_list, logits_list = [], []
    for s in range(0, len(wins), BATCH_WINDOWS):
        chunk = np.stack(wins[s: s + BATCH_WINDOWS]).astype(np.float32)
        outs = sess.run(None, {in_name: chunk})
        embs_list.append(outs[out_map.get("embedding", 1)].astype(np.float32))
        logits_list.append(outs[out_map.get("label", 0)].astype(np.float32))
        del chunk
    return np.concatenate(embs_list, 0), np.concatenate(logits_list, 0)


def main():
    print(f"Versão = {DATA_VERSION}")
    df = readCSV(pathCSV)
    if df is None:
        print("Dataframe não encontrado!"); return
    print(f"Total de Linhas no CSV = {len(df)}")

    df = filter_df(df)
    print(f"Linhas após filtros = {len(df)}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(PERCH_ONNX_PATH), sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_map = {o.name: i for i, o in enumerate(sess.get_outputs())}
    print(f"Perch ONNX: {PERCH_ONNX_PATH.name}")

    # Probe para descobrir dimensões
    dummy = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    outs = sess.run(None, {in_name: dummy})
    emb_dim = outs[out_map.get("embedding", 1)].shape[1]
    n_bc    = outs[out_map.get("label", 0)].shape[1]
    del dummy, outs
    print(f"  emb_dim={emb_dim}, n_bc={n_bc}")

    # ─── Streaming: 1 ROI por vez ──────────────────────────────────
    print("\nProcessando ROI por ROI (streaming)...")
    rows_feat = []
    rows_meta = []
    n_falhas  = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        seg = load_segment(row["soundscape_file"], row["roi_start"], row["roi_end"])
        if seg is None:
            n_falhas += 1
            continue

        wins = segment_to_windows(seg)
        del seg
        if not wins:
            n_falhas += 1
            continue

        embs, logits = infer_windows(sess, in_name, out_map, wins)
        del wins
        feat = aggregate_windows(embs, logits, AGG_MODE)
        del embs, logits

        rows_feat.append(feat)
        rows_meta.append({
            "audioSource": row["soundscape_file"],
            "roi_label":   row["roi_label"],
            "min_freq":    float(row["roi_min_freq"]),
            "max_freq":    float(row["roi_max_freq"]),
            "duration":    float(row["roi_duration"]),
        })

        if (i + 1) % 1000 == 0:
            gc.collect()

    print(f"  ROIs processados: {len(rows_feat)} | falhas: {n_falhas}")

    if not rows_feat:
        print("Nada para salvar."); return

    if AGG_MODE == "mean_std":
        emb_cols   = [f"perch_emb_mean_{j}" for j in range(emb_dim)] + \
                     [f"perch_emb_std_{j}"  for j in range(emb_dim)]
        logit_cols = [f"perch_logit_mean_{j}" for j in range(n_bc)] + \
                     [f"perch_logit_max_{j}"  for j in range(n_bc)]
    else:
        emb_cols   = [f"perch_emb_{j}"   for j in range(emb_dim)]
        logit_cols = [f"perch_logit_{j}" for j in range(n_bc)]

    feat_df = pd.DataFrame(np.stack(rows_feat), columns=emb_cols + logit_cols)
    meta_df = pd.DataFrame(rows_meta)
    dfCut   = pd.concat([meta_df, feat_df], axis=1)

    bad = dfCut[emb_cols + logit_cols].isna().any(axis=1) | \
          np.isinf(dfCut[emb_cols + logit_cols].to_numpy()).any(axis=1)
    if bad.any():
        print(f"  descartando {int(bad.sum())} linhas com NaN/inf")
        dfCut = dfCut[~bad].reset_index(drop=True)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(dfCut, f)

    print(f"\n Salvo: {OUTPUT_PKL}")
    print(f"   shape: {dfCut.shape}")
    print(dfCut.iloc[:3, :6].to_string())


if __name__ == "__main__":
    main()