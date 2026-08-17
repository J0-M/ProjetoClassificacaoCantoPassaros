import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_VERSION = "v4_novas_features"
CV_SPLITS = [2, 3, 5, 10]
RANDOM_STATE = 1


def salvar_objeto(obj, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(obj, f)

def preparar_dataframe(df, n_splits):
    df = df.dropna(subset=["roi_label"]).copy()
    df["roi_label"] = df["roi_label"].astype(str)

    # counts = df["roi_label"].value_counts()
    # classes_validas = counts[counts >= CV_SPLITS].index
    # df = df[df["roi_label"].isin(classes_validas)].copy()

    grupos_por_classe = df.groupby("roi_label")["audioSource"].nunique()
    classes_validas_grupos = grupos_por_classe[grupos_por_classe >= n_splits].index
    df = df[df["roi_label"].isin(classes_validas_grupos)].copy()

    df = df.reset_index(drop=True)

    return df


def gerar_folds(df, output_path, n_splits):

    df = preparar_dataframe(df, n_splits)

    print(f"\n===== {n_splits} FOLDS =====")
    print(f"Total amostras após filtro: {len(df)}")
    print(f"Total classes após filtro: {df['roi_label'].nunique()}")
    
    print("\nÁudios distintos por espécie:")
    print(
        df.groupby("roi_label")["audioSource"]
        .nunique()
        .describe()
    )
    
    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]
    groups = df["audioSource"]

    skf = StratifiedGroupKFold(
        n_splits=n_splits,
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

    for n_splits in CV_SPLITS:   
        output_path = (f"{DATA_VERSION}/{tipo}/stratified_group_kfold_{n_splits}.pkl")
        gerar_folds(df, output_path, n_splits)

    print("Folds salvos com sucesso!")

if __name__ == "__main__":
    main()
