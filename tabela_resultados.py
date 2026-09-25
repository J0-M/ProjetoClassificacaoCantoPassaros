import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, top_k_accuracy_score

# ============================================
# Configurações
# ============================================

DATA_VERSIONS = ["v1_media", "v2_media_std", "v3_media_std_freq",
                 "v4_novas_features", "v5_novo_filtro", "v6_perch"]
TYPES = ["segmentado", "completo"]

CLASSIFIERS = {
    "KNN":         "knn",
    "SVM":         "svm",
    "XGBoost":     "xgboost",
    "KMeansC_NC":  "kmeansc_nc",
    "KMeansC_SVM": "kmeansc_svm",
    "KMeansD":     "kmeansd",
}

N_SPLITS = [5, 10]
TOP_K = 5

CSV_OUT = "tabela_comparativa_global.csv"

# ============================================
# Utilitários
# ============================================

def carregar_objeto(caminho):
    with open(caminho, "rb") as f:
        return pickle.load(f)

def calcular_metricas(y_true, y_proba, classes, k=TOP_K):
    y_true = np.asarray(y_true)
    mask = np.isin(y_true, classes)
    y_true_f = y_true[mask]
    y_proba_f = y_proba[mask]
    if len(y_true_f) == 0:
        return None
    y_pred = classes[np.argmax(y_proba_f, axis=1)]
    f1 = f1_score(y_true_f, y_pred, average="macro")
    topk = top_k_accuracy_score(y_true_f, y_proba_f, k=k, labels=classes)
    return f1, topk

def metricas_fold(caminho_matriz):
    try:
        m = carregar_objeto(caminho_matriz)
        return calcular_metricas(m["y_true"], m["y_proba"], m["classes"])
    except Exception:
        return None

def caminho_base(nome_clf, sufixo, version, tipo, n_splits):
    pasta_clf = "KMeansC_SVM" if nome_clf == "KMeansD" else nome_clf
    return os.path.join(
        pasta_clf, version,
        f"matrizesProba_{sufixo}_treino{tipo.capitalize()}",
        f"{n_splits}fold",
    )

# ============================================
# Coleta
# ============================================

def coletar_resultados():
    registros = []
    total = len(DATA_VERSIONS) * len(TYPES) * len(CLASSIFIERS) * len(N_SPLITS)
    feito = 0

    for version in DATA_VERSIONS:
        for tipo in TYPES:
            for nome_clf, sufixo in CLASSIFIERS.items():
                for n_splits in N_SPLITS:
                    feito += 1
                    prog = f"[{feito:>3}/{total}]"
                    base = caminho_base(nome_clf, sufixo, version, tipo, n_splits)
                    if not os.path.isdir(base):
                        continue

                    f1_list, topk_list = [], []
                    for fold_id in range(1, n_splits + 1):
                        arq = os.path.join(base, f"matriz_{fold_id}.pkl")
                        if not os.path.exists(arq):
                            continue
                        met = metricas_fold(arq)
                        if met is None:
                            continue
                        f1, topk = met
                        f1_list.append(f1)
                        topk_list.append(topk)

                    if not f1_list:
                        continue

                    registros.append({
                        "version":    version,
                        "tipo":       tipo,
                        "classifier": nome_clf,
                        "n_splits":   n_splits,
                        "n_folds_ok": len(f1_list),
                        "f1_mean":    float(np.mean(f1_list)),
                        "f1_std":     float(np.std(f1_list)),
                        "topk_mean":  float(np.mean(topk_list)),
                        "topk_std":   float(np.std(topk_list)),
                    })
                    print(f"{prog} {version:>18} | {tipo:<10} | {nome_clf:<12} | "
                          f"{n_splits}-fold : "
                          f"F1={np.mean(f1_list):.4f}±{np.std(f1_list):.4f} | "
                          f"Top-{TOP_K}={np.mean(topk_list):.4f}±{np.std(topk_list):.4f}")

    return pd.DataFrame(registros)

# ============================================
# Tabela por (versão, tipo) — formato original
# ============================================

def montar_tabela(sub):
    """
    sub: DataFrame filtrado por (version, tipo).
    Retorna DataFrame com uma linha por classificador e colunas:
    Classificador | {n}-Fold F1 | {n}-Fold Top-K | ... para cada n em N_SPLITS
    """
    linhas = []
    for nome_clf in CLASSIFIERS.keys():
        linha = {"Classificador": nome_clf}
        for n_splits in N_SPLITS:
            r = sub[(sub["classifier"] == nome_clf) & (sub["n_splits"] == n_splits)]
            if r.empty:
                linha[f"{n_splits}-Fold F1"]       = "N/D"
                linha[f"{n_splits}-Fold Top-{TOP_K}"] = "N/D"
            else:
                f1m = r["f1_mean"].iloc[0];  f1s = r["f1_std"].iloc[0]
                tkm = r["topk_mean"].iloc[0]; tks = r["topk_std"].iloc[0]
                linha[f"{n_splits}-Fold F1"]       = f"{f1m:.4f}±{f1s:.4f}"
                linha[f"{n_splits}-Fold Top-{TOP_K}"] = f"{tkm:.4f}±{tks:.4f}"
        linhas.append(linha)
    return pd.DataFrame(linhas)

def largura_colunas(tabela):
    """Larguras dinâmicas, baseadas no cabeçalho e nos valores."""
    larguras = {}
    for col in tabela.columns:
        w = max(len(col), tabela[col].astype(str).str.len().max())
        larguras[col] = w + 2
    return larguras

def imprimir_tabela_original(df):
    """Uma tabela por (versão, tipo), no mesmo layout do script original."""
    combos = (df[["version", "tipo"]]
              .drop_duplicates()
              .sort_values(["version", "tipo"]))

    for _, row in combos.iterrows():
        version, tipo = row["version"], row["tipo"]
        sub = df[(df["version"] == version) & (df["tipo"] == tipo)]
        if sub.empty:
            continue

        tabela = montar_tabela(sub)
        larg = largura_colunas(tabela)

        titulo = f"Resultados para versão '{version}' - tipo '{tipo}'"
        total_w = sum(larg.values())

        print()
        print("═" * total_w)
        print(f"  {titulo}")
        print("═" * total_w)

        # Cabeçalho
        header = "".join(f"{col:<{larg[col]}}" for col in tabela.columns)
        print(header)
        print("-" * total_w)

        # Linhas
        for _, linha in tabela.iterrows():
            print("".join(f"{str(linha[col]):<{larg[col]}}" for col in tabela.columns))

# ============================================
# Main
# ============================================

def main():
    print(f"Varredura: {len(DATA_VERSIONS)} versões × "
          f"{len(TYPES)} tipos × {len(CLASSIFIERS)} classificadores × "
          f"{len(N_SPLITS)} splits")
    print("─" * 130)

    df = coletar_resultados()
    if df.empty:
        print("Nada encontrado. Confira se os diretórios matrizesProba_* existem.")
        return

    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"\nCSV salvo: {CSV_OUT}  ({len(df)} linhas)")

    # ─── Tabelas no formato original, uma por (versão, tipo) ───
    imprimir_tabela_original(df)

    # ─── Ranking por n_splits ───
    print()
    print("═" * 130)
    print("  MELHORES COMBINAÇÕES POR n_splits  (F1 Macro)")
    print("═" * 130)
    for n in N_SPLITS:
        sub = df[df["n_splits"] == n].sort_values("f1_mean", ascending=False)
        if sub.empty:
            continue
        top5 = sub.head(5)[["version", "tipo", "classifier", "f1_mean", "topk_mean"]]
        print(f"\n--- {n}-fold ---")
        print(top5.to_string(index=False))

if __name__ == "__main__":
    main()