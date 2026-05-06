import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

############################################
# CONFIG
############################################

DATA_VERSION = "v4_novas_features"
TIPO = "segmentado"
FOLD_IDX = 0
CV_SPLITS = 10
RANDOM_STATE = 1

############################################

def carregar(path):
    with open(path, "rb") as f:
        return pickle.load(f)

############################################
# PRÉ-PROCESSAMENTO (IDÊNTICO AO ORIGINAL)
############################################

def preparar_dataframe(df):
    df = df.dropna(subset=["roi_label"]).copy()
    df["roi_label"] = df["roi_label"].astype(str)

    counts = df["roi_label"].value_counts()
    classes_validas = counts[counts >= CV_SPLITS].index
    df = df[df["roi_label"].isin(classes_validas)].copy()

    grupos_por_classe = df.groupby("roi_label")["audioSource"].nunique()
    classes_validas_grupos = grupos_por_classe[grupos_por_classe >= 2].index
    df = df[df["roi_label"].isin(classes_validas_grupos)].copy()

    df = df.reset_index(drop=True)

    return df

############################################
# TESTE 1 — Vazamento de grupos (fold atual)
############################################

def teste_grupos(df, fold):
    train_groups = set(df.iloc[fold["train_idx"]]["audioSource"])
    test_groups = set(df.iloc[fold["test_idx"]]["audioSource"])

    intersec = train_groups.intersection(test_groups)

    print("\n[TESTE 1] Vazamento de grupos (fold)")
    print("Interseção:", len(intersec))

    if intersec:
        print("LEAKAGE REAL DETECTADO")
    else:
        print("Sem vazamento de grupos")

############################################
# TESTE 2 — Vazamento GLOBAL entre folds
############################################

def teste_grupos_global(df, folds):
    print("\n[TESTE 2] Grupos em múltiplos folds (global)")

    grupo_para_folds = {}

    for i, fold in enumerate(folds):
        grupos = df.iloc[fold["test_idx"]]["audioSource"].unique()

        for g in grupos:
            grupo_para_folds.setdefault(g, set()).add(i)

    multi = {g: f for g, f in grupo_para_folds.items() if len(f) > 1}

    print("Grupos que aparecem em mais de um fold:", len(multi))

############################################
# TESTE 3 — Similaridade
############################################

def teste_similaridade(X_train, X_test):
    print("\n[TESTE 3] Similaridade treino vs teste")

    n_tr = min(1000, len(X_train))
    n_te = min(1000, len(X_test))

    idx_tr = np.random.choice(len(X_train), n_tr, replace=False)
    idx_te = np.random.choice(len(X_test), n_te, replace=False)

    sim = cosine_similarity(X_train[idx_tr], X_test[idx_te])

    print("Similaridade máxima:", sim.max())
    print("Similaridade média:", sim.mean())

############################################
# TESTE 4 — Duplicatas exatas
############################################

def teste_duplicatas(df_train, df_test):
    print("\n[TESTE 4] Duplicatas exatas entre treino e teste")

    cols = [c for c in df_train.columns if c not in ["roi_label", "audioSource"]]

    merged = pd.merge(
        df_train[cols],
        df_test[cols],
        on=cols,
        how="inner"
    )

    print("Duplicatas encontradas:", len(merged))

    if len(merged) > 0:
        print("POSSÍVEL LEAKAGE (duplicatas)")
    else:
        print("Sem duplicatas")

############################################
# TESTE 5 — Label shuffle
############################################

def teste_label_shuffle(X_tr, X_val, y_tr, y_val):
    print("\n[TESTE 5] Label shuffle")

    y_random = np.random.permutation(y_tr)

    svm = SVC(C=100, gamma="scale")
    svm.fit(X_tr, y_random)

    pred = svm.predict(X_val)
    f1 = f1_score(y_val, pred, average="macro")

    print("F1 (aleatório):", f1)

############################################
# TESTE 6 — Modelo simples
############################################

def teste_modelo_simples(X_tr, X_val, y_tr, y_val):
    print("\n[TESTE 6] KNN baseline")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_tr, y_tr)

    pred = knn.predict(X_val)
    f1 = f1_score(y_val, pred, average="macro")

    print("F1 (KNN):", f1)

############################################

def main():

    print("=== DIAGNÓSTICO DE LEAKAGE ===")

    df_path = f"dataframes/{DATA_VERSION}/dataframeSegmentado.pkl"
    folds_path = f"folds/{DATA_VERSION}/{TIPO}/stratified_group_kfold_10.pkl"

    df = carregar(df_path)
    folds = carregar(folds_path)

    df = preparar_dataframe(df)

    ########################################
    # TESTES GLOBAIS
    ########################################

    teste_grupos_global(df, folds)

    ########################################
    # SELECIONA FOLD
    ########################################

    fold = folds[FOLD_IDX]

    df_train = df.iloc[fold["train_idx"]]
    df_test = df.iloc[fold["test_idx"]]

    ########################################
    # TESTES
    ########################################

    teste_grupos(df, fold)
    teste_duplicatas(df_train, df_test)

    ########################################
    # FEATURES
    ########################################

    X_train = df_train.drop(columns=["roi_label", "audioSource"]).values
    y_train = df_train["roi_label"]

    X_test = df_test.drop(columns=["roi_label", "audioSource"]).values
    y_test = df_test["roi_label"]

    ########################################
    # SCALER
    ########################################

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ########################################
    teste_similaridade(X_train, X_test)

    ########################################
    # SPLIT INTERNO (CORRIGIDO)
    ########################################

    counts = y_train.value_counts()
    classes_validas = counts[counts >= 2].index

    mask = y_train.isin(classes_validas).values  # 🔥 CORREÇÃO
    X_train = X_train[mask]
    y_train = y_train[mask]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    ########################################
    teste_label_shuffle(X_tr, X_val, y_tr, y_val)
    teste_modelo_simples(X_tr, X_val, y_tr, y_val)

############################################

if __name__ == "__main__":
    main()