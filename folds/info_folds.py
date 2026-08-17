import os
import pickle
import logging
import pandas as pd

DATA_VERSION = "v4_novas_features"
CV_SPLITS = [2, 3, 5, 10]
DF_PATH = f"../dataframes/{DATA_VERSION}/dataframeSegmentado.pkl"
OUTPUT_DIR = f"{DATA_VERSION}/segmentado"

def preparar_dataframe(df, n_splits):

    df = df.dropna(subset=["roi_label"]).copy()
    df["roi_label"] = df["roi_label"].astype(str)

    grupos_por_classe = (df.groupby("roi_label")["audioSource"].nunique())
    classes_validas_grupos = grupos_por_classe[grupos_por_classe >= n_splits].index

    df = df[df["roi_label"].isin(classes_validas_grupos)].copy()
    df = df.reset_index(drop=True)

    return df


def configurar_log(log_path):
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(f"folds_{os.path.basename(log_path)}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        
        handler = logging.FileHandler(log_path,encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def processar_split(df_original, n_splits):

    print()
    print("=" * 80)
    print(f"PROCESSANDO {n_splits}-FOLD")
    print("=" * 80)

    folds_path = os.path.join(OUTPUT_DIR, f"stratified_group_kfold_{n_splits}.pkl")
    log_path = os.path.join(OUTPUT_DIR, f"folds_amostras_{n_splits}fold.log")
    csv_path = os.path.join(OUTPUT_DIR, f"folds_amostras_{n_splits}fold.csv")
    logger = configurar_log(log_path)

    df = preparar_dataframe(
        df_original,
        n_splits
    )

    print(f"Amostras após filtro: {len(df)}")
    print(f"Espécies: {df['roi_label'].nunique()}")

    logger.info("=" * 80)
    logger.info(f"REGISTRO DOS FOLDS - {n_splits}-FOLD")
    logger.info("=" * 80)

    logger.info(f"DATA_VERSION: {DATA_VERSION}")
    logger.info(f"n_splits: {n_splits}")
    logger.info(f"Total de amostras: {len(df)}")
    logger.info(f"Total de espécies: {df['roi_label'].nunique()}")

    if not os.path.exists(folds_path):

        logger.error(f"Arquivo de folds não encontrado: {folds_path}")
        print(f"ERRO: folds não encontrados: {folds_path}")

        return

    print(f"Carregando folds: stratified_group_kfold_{n_splits}.pkl")

    with open(folds_path, "rb") as f:
        folds = pickle.load(f)

    registros = []

    for fold_info in folds:

        fold_id = fold_info["fold"]

        train_idx = fold_info["train_idx"]
        test_idx = fold_info["test_idx"]

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            f"FOLD {fold_id}"
        )
        logger.info("=" * 80)

        logger.info(f"Treino: {len(train_idx)} amostras")
        logger.info(f"Teste: {len(test_idx)} amostras")

        train_audios = set(df.iloc[train_idx]["audioSource"])
        test_audios = set(df.iloc[test_idx]["audioSource"])
        intersecao = (train_audios & test_audios)

        if intersecao:
            logger.error(f"FOLD {fold_id}: VAZAMENTO DE GRUPOS! {len(intersecao)} áudios aparecem em treino e teste.")
            logger.error(f"Áudios problemáticos: {intersecao}")

        else:
            logger.info(f"FOLD {fold_id}: OK - nenhum audioSource compartilhado entre treino e teste.")

        logger.info("")
        logger.info("--- TREINO ---")

        for idx in train_idx:

            row = df.iloc[idx]

            especie = row["roi_label"]
            audio = row["audioSource"]

            logger.info(f"idx={idx} | especie={especie} | audio={audio}")

            registros.append({
                "n_splits": n_splits,
                "fold": fold_id,
                "conjunto": "train",
                "indice": idx,
                "especie": especie,
                "audioSource": audio
            })

        logger.info("")
        logger.info("--- TESTE ---")

        for idx in test_idx:

            row = df.iloc[idx]

            especie = row["roi_label"]
            audio = row["audioSource"]

            logger.info(
                f"idx={idx} | "
                f"especie={especie} | "
                f"audio={audio}"
            )

            registros.append({
                "n_splits": n_splits,
                "fold": fold_id,
                "conjunto": "test",
                "indice": idx,
                "especie": especie,
                "audioSource": audio
            })

    df_registros = pd.DataFrame(registros)
    
    df_registros.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Log: {log_path}")
    print(f"CSV: {csv_path}")
    print(f"Total de registros: {len(df_registros)}")

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def main():

    print(f"DATA_VERSION: {DATA_VERSION}")

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    print(f"Dataframe original: {len(df)} amostras")
    
    for n_splits in CV_SPLITS:
        processar_split(df, n_splits)


if __name__ == "__main__":
    main()