import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_VERSION = "v3_media_std_freq"
CV_SPLITS = 10
RANDOM_STATE = 1


def salvar_objeto(obj, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(obj, f)


def gerar_folds(df, output_path):

    df = df.dropna(subset=["roi_label"]).copy()
    df["roi_label"] = df["roi_label"].astype(str)
    
    y = df["roi_label"]

    counts = y.value_counts()
    classes_validas = counts[counts >= CV_SPLITS].index

    df = df[df["roi_label"].isin(classes_validas)]
    
    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]
    groups = df["audioSource"]

    skf = StratifiedGroupKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        folds.append({
            "fold": fold_id,
            "train_idx": train_idx,
            "test_idx": test_idx
        })

    salvar_objeto(folds, output_path)


def main():

    print(f"== VERSÃO: {DATA_VERSION} ==")

    tipo = input("1 - Segmentado\n2 - Completo\nEscolha: ").strip()
    tipo = {"1": "segmentado", "2": "completo"}[tipo]

    df_path = f"../dataframes/{DATA_VERSION}/dataframe{'Segmentado' if tipo=='segmentado' else 'AudioCompleto'}.pkl"
    df = pickle.load(open(df_path, "rb"))

    output_path = f"{DATA_VERSION}/{tipo}/stratified_group_kfold_{CV_SPLITS}.pkl"

    gerar_folds(df, output_path)

    print("Folds salvos com sucesso!")


if __name__ == "__main__":
    main()
