import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_VERSION = "v4_novas_features"

CV_SPLITS = [3, 5, 10]
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
            "X_train": X.iloc[train_idx].copy(),
            "y_train": y.iloc[train_idx].copy(),
            "X_test": X.iloc[test_idx].copy(),
            "y_test": y.iloc[test_idx].copy()
        })

    salvar_objeto(folds, output_path)
    

def gerar_folds_novo_filtro(df, input_path, output_path, n_splits):
    
    if not os.path.isdir(input_path):
        print(f"[AVISO] Pasta não encontrada: {input_path}")
        return None
    
    df = df.copy()
    df["_key"] = df["audioSource"].astype(str) + "||" + df["roi_label"].astype(str)
    key_to_idx = {k: i for i, k in enumerate(df["_key"])}
    
    def ler_txt(caminho):
        chaves = []
        with open(caminho, "r", encoding="utf-8-sig") as f:
            for linha in f:
                linha = linha.strip()
                if not linha or "," not in linha:
                    continue
                # rsplit(",", 1) para preservar vírgulas no rótulo, se houver
                audio, label = linha.rsplit(",", 1)
                chaves.append(f"{audio.strip()}||{label.strip()}")
        return chaves
    
    folds = []

    for fold_id in range(1, n_splits + 1):
        test_path = os.path.join(input_path, f"fold_{fold_id}_test.txt")
        train_path = os.path.join(input_path, f"fold_{fold_id}_training.txt")

        if not (os.path.exists(test_path) and os.path.exists(train_path)):
            print(f"[AVISO] Fold {fold_id}: txts não encontrados em {input_path}, pulando.")
            continue

        test_keys = set(ler_txt(test_path))
        train_keys = set(ler_txt(train_path))

        test_idx = [key_to_idx[k] for k in test_keys if k in key_to_idx]
        train_idx = [key_to_idx[k] for k in train_keys if k in key_to_idx]

        # Log de quantas chaves dos txts não bateram com o dataframe
        n_falta_test = len(test_keys) - len(test_idx)
        n_falta_train = len(train_keys) - len(train_idx)
        if n_falta_test or n_falta_train:
            print(f"[AVISO] Fold {fold_id}: {n_falta_test} chaves teste e "
                  f"{n_falta_train} chaves treino não existem no dataframe.")

        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        X_train = df_train.drop(columns=["roi_label", "audioSource", "_key"]).reset_index(drop=True)
        y_train = df_train["roi_label"].reset_index(drop=True)
        X_test = df_test.drop(columns=["roi_label", "audioSource", "_key"]).reset_index(drop=True)
        y_test = df_test["roi_label"].reset_index(drop=True)

        folds.append({
            "fold": fold_id - 1,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        })

        print(f"Fold {fold_id}: treino={len(X_train)} | teste={len(X_test)}")

    # Salva no mesmo formato dos pkls de v4_novas_features/<tipo>/
    salvar_objeto(folds, output_path)
    print(f"Folds ({n_splits}) salvos em: {output_path}")

    return folds


def main():

    print(f"== VERSÃO: {DATA_VERSION} ==")

    modo = input("1 - Folds estratificados (v4)\n2 - Folds a partir de novo_filtro (txts)\nEscolha: ").strip()
    tipo = input("1 - Segmentado\n2 - Completo\nEscolha: ").strip()
    tipo = {"1": "segmentado", "2": "completo"}[tipo]

    df_path = f"../dataframes/{DATA_VERSION}/dataframe{'Segmentado' if tipo == 'segmentado' else 'AudioCompleto'}.pkl"
    df = pickle.load(open(df_path, "rb"))

    if modo == "1":
        for n_splits in CV_SPLITS:
            output_path = f"{DATA_VERSION}/{tipo}/stratified_group_kfold_{n_splits}.pkl"
            gerar_folds(df, output_path, n_splits)

    elif modo == "2":
        for n_splits in CV_SPLITS:
            input_path  = f"novo_filtro/k-{n_splits}"
            output_path = f"novo_filtro/{tipo}/stratified_group_kfold_{n_splits}.pkl"
            gerar_folds_novo_filtro(df, input_path, output_path, n_splits)

    print("Folds salvos com sucesso!")

if __name__ == "__main__":
    main()
